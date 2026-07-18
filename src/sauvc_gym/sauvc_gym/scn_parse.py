"""
Introspection of a Stonefish scenario (.scn) file to recover thruster geometry.

Design rationale
----------------
The plant lives in the scene file. The controller must not carry a second,
hand-copied idea of where the thrusters are -- that is exactly how sim and real
drift apart. So instead of hardcoding an allocation matrix, we read the
``<actuator type="thruster">`` blocks out of the scenario the simulator is
actually running, and build the allocation matrix from those numbers.

If you move a thruster in the .scn, the allocation follows automatically.

Stonefish conventions this module relies on
-------------------------------------------
* Body frame is NED: +X forward, +Y starboard, +Z down.
* A thruster produces force along the **local +X axis** of its own frame,
  which is set by ``<origin xyz="..." rpy="..."/>`` relative to the link.
* ``T = Kt * w * |w|`` with ``w`` in rad/s (validated empirically for this
  vehicle: Kt = 0.0005, w_max ~ 314 rad/s -> ~49 N).
* A left-handed propeller (``right="false"``) reverses the sign of the
  produced thrust; ``inverted_setpoint="true"`` reverses the sign of the
  incoming setpoint. The two together cancel.

The last point is a *convention we assume*, not one we trust: the sign of every
column is meant to be confirmed empirically with ``scripts/identify_allocation.py``.
See README section "Trust, but identify".
"""

from __future__ import annotations

import math
import os
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

__all__ = ["ThrusterSpec", "VehicleSpec", "parse_scenario", "rpy_to_matrix"]


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Fixed-axis XYZ (roll-pitch-yaw) to rotation matrix, R = Rz @ Ry @ Rx.

    This matches the URDF/Stonefish ``rpy`` convention.
    """
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return rz @ ry @ rx


@dataclass
class ThrusterSpec:
    """One thruster, as declared in the scenario file."""

    name: str
    position: np.ndarray  # (3,) body-frame origin [m]
    direction: np.ndarray  # (3,) unit thrust axis in body frame (local +X)
    thrust_coeff: float  # Kt in T = Kt * w * |w|
    max_omega: float  # rad/s at |setpoint| = 1
    right_handed: bool  # <propeller right="true|false">
    inverted_setpoint: bool  # <thruster inverted_setpoint="true|false">

    @property
    def max_thrust(self) -> float:
        """Thrust magnitude at |setpoint| = 1, in Newtons."""
        return self.thrust_coeff * self.max_omega**2

    @property
    def setpoint_sign(self) -> float:
        """Sign mapping a positive setpoint onto thrust along ``direction``.

        A left-handed propeller flips the thrust sign; ``inverted_setpoint``
        flips the setpoint sign. Both set -> they cancel -> +1.
        Exactly one set -> -1.
        """
        flips = (not self.right_handed) ^ self.inverted_setpoint
        return -1.0 if flips else 1.0

    def thrust_from_setpoint(self, u: float) -> float:
        """Signed thrust [N] along ``direction`` for a setpoint u in [-1, 1].

        Quadratic, because the underlying model is quadratic in shaft speed and
        the setpoint is proportional to shaft speed:
            w = u * w_max
            T = Kt * w * |w| = Kt * w_max^2 * u * |u| = T_max * u * |u|
        """
        return self.setpoint_sign * self.max_thrust * u * abs(u)

    def setpoint_from_thrust(self, t: float) -> float:
        """Inverse of :meth:`thrust_from_setpoint`, clipped to [-1, 1]."""
        t_signed = t * self.setpoint_sign
        u = math.copysign(math.sqrt(abs(t_signed) / self.max_thrust), t_signed)
        return float(np.clip(u, -1.0, 1.0))


@dataclass
class VehicleSpec:
    """The parts of a scenario a controller actually needs."""

    robot_name: str
    thrusters: list[ThrusterSpec] = field(default_factory=list)
    setpoint_topic: str | None = None
    source_file: str | None = None

    @property
    def n_thrusters(self) -> int:
        return len(self.thrusters)

    @property
    def names(self) -> list[str]:
        return [t.name for t in self.thrusters]

    @property
    def max_thrusts(self) -> np.ndarray:
        return np.array([t.max_thrust for t in self.thrusters], dtype=float)

    def allocation_matrix(self) -> np.ndarray:
        """6 x N matrix B such that ``wrench = B @ thrusts``.

        Rows are ``[Fx, Fy, Fz, Mx, My, Mz]`` in the body frame; ``thrusts`` is
        the vector of signed thrust magnitudes along each thruster's own axis.

        Note this maps *thrust*, not *setpoint*, onto wrench -- the setpoint
        relationship is quadratic and is handled separately.
        """
        b = np.zeros((6, self.n_thrusters), dtype=float)
        for i, t in enumerate(self.thrusters):
            b[0:3, i] = t.direction
            b[3:6, i] = np.cross(t.position, t.direction)
        return b


def _resolve_ros_paths(text: str) -> str:
    """Neutralise ``$(find pkg)`` so ElementTree can parse the file."""
    return re.sub(r"\$\(find\s+([^)]+)\)", r"__FIND_\1__", text)


def _substitute_args(text: str, args: dict[str, str]) -> str:
    """Resolve ``$(param name)`` / ``$(arg name)`` against supplied values."""
    def repl(m: re.Match) -> str:
        key = m.group(2).strip()
        return str(args.get(key, m.group(0)))

    return re.sub(r"\$\((param|arg)\s+([^)]+)\)", repl, text)


def _to_float_vec(s: str, n: int = 3) -> np.ndarray:
    parts = [p for p in re.split(r"[\s,]+", s.strip()) if p]
    if len(parts) != n:
        raise ValueError(f"expected {n} numbers, got {s!r}")
    return np.array([float(p) for p in parts], dtype=float)


def _bool_attr(elem: ET.Element | None, name: str, default: bool) -> bool:
    if elem is None or elem.get(name) is None:
        return default
    return str(elem.get(name)).strip().lower() in ("true", "1", "yes")


def parse_scenario(
    path: str,
    robot_name: str | None = None,
    args: dict[str, str] | None = None,
    default_max_omega: float = 314.0,
) -> VehicleSpec:
    """Read thruster geometry out of a Stonefish scenario file.

    Parameters
    ----------
    path:
        Path to the ``.scn`` containing the ``<robot>`` definition. Note this
        should be the *vehicle* scenario, not the arena scenario that merely
        ``<include>``s it -- includes are not followed.
    robot_name:
        Which robot to read, if the file defines several. Defaults to the first.
    args:
        Values for ``$(param x)`` / ``$(arg x)`` placeholders, e.g.
        ``{"robot_name": "sauvc_auv"}``.
    default_max_omega:
        Shaft speed at |setpoint| = 1 when the scenario does not state one.

    Raises
    ------
    FileNotFoundError, ValueError
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as fh:
        raw = fh.read()
    raw = _substitute_args(_resolve_ros_paths(raw), args or {})

    try:
        root = ET.fromstring(raw)
    except ET.ParseError as exc:
        raise ValueError(f"{path}: not parseable as XML ({exc}). ") from exc

    robots = root.findall(".//robot")
    if not robots:
        raise ValueError(
            f"{path}: no <robot> element. If this is the arena scenario, point "
            f"at the vehicle scenario it includes instead -- includes are not followed."
        )

    robot = robots[0]
    if robot_name is not None:
        match = [r for r in robots if r.get("name") == robot_name]
        if not match:
            found = [r.get("name") for r in robots]
            raise ValueError(f"{path}: robot {robot_name!r} not found; have {found}")
        robot = match[0]

    spec = VehicleSpec(robot_name=robot.get("name", "robot"), source_file=path)

    sub = robot.find("ros_subscriber")
    if sub is not None:
        spec.setpoint_topic = sub.get("thrusters")

    for act in robot.findall(".//actuator"):
        if act.get("type") != "thruster":
            continue

        origin = act.find("origin")
        if origin is None:
            raise ValueError(f"{path}: thruster {act.get('name')!r} has no <origin>")
        pos = _to_float_vec(origin.get("xyz", "0 0 0"))
        rpy = _to_float_vec(origin.get("rpy", "0 0 0"))
        direction = rpy_to_matrix(*rpy) @ np.array([1.0, 0.0, 0.0])

        prop = act.find("propeller")
        right = _bool_attr(prop, "right", True)

        # Kt lives under <thrust_model><coeff .../></thrust_model> or as an
        # attribute; accept several spellings rather than guess one.
        kt = None
        for tag in ("thrust_model", "rotor_dynamics", "propeller"):
            node = act.find(tag)
            if node is None:
                continue
            for attr in ("thrust_coeff", "coeff", "kt"):
                if node.get(attr) is not None:
                    kt = float(node.get(attr))
                    break
            if kt is None:
                child = node.find("thrust_coeff")
                if child is not None and child.get("value") is not None:
                    kt = float(child.get("value"))
            if kt is not None:
                break
        if kt is None:
            raise ValueError(
                f"{path}: could not find a thrust coefficient for thruster "
                f"{act.get('name')!r}. Pass it explicitly or check the tag name "
                f"against your Stonefish version."
            )

        max_omega = default_max_omega
        for tag in ("rotor_dynamics", "thrust_model", "propeller"):
            node = act.find(tag)
            if node is not None and node.get("max_omega") is not None:
                max_omega = float(node.get("max_omega"))
                break

        spec.thrusters.append(
            ThrusterSpec(
                name=act.get("name", f"thruster_{len(spec.thrusters)}"),
                position=pos,
                direction=direction / np.linalg.norm(direction),
                thrust_coeff=kt,
                max_omega=max_omega,
                right_handed=right,
                inverted_setpoint=_bool_attr(act, "inverted_setpoint", False),
            )
        )

    if not spec.thrusters:
        raise ValueError(f"{path}: robot {spec.robot_name!r} declares no thrusters")

    return spec
