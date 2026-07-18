"""
Thrust allocation for the vectored 8-thruster SAUVC vehicle.

Why the agent does not command thrusters directly
--------------------------------------------------
The obvious action space is 8 raw setpoints in [-1, 1]. It is also the wrong
one, for four reasons:

1. **It is over-parameterised.** The vehicle has 8 thrusters but the horizontal
   four span only a 3-D wrench subspace (Fx, Fy, Mz) and the vertical four span
   (Fz, Mx, My). A policy in R^8 spends its early samples rediscovering a
   4-dimensional nullspace that we already know analytically.
2. **It relearns known physics.** The geometry is in the scene file. Making the
   policy infer it from reward is paying sample cost for a matrix we can write down.
3. **It fights the sign conventions.** Left-handed propellers, inverted
   setpoints -- an RL policy will happily learn around a sign error and then
   fail on hardware, silently.
4. **It does not match the deployment path.** On the real vehicle the Pixhawk
   runs ArduSub, which takes a *wrench-like* command (surge/sway/heave/yaw) and
   does its own allocation. A policy trained on raw Stonefish thruster indices
   has no clean place to plug in.

So the action is a normalised wrench, ``a in [-1, 1]^4`` over
(surge, sway, heave, yaw), and this module turns it into setpoints. That keeps
the policy in a physically meaningful, low-dimensional, hardware-portable space,
and leaves the allocation as a fixed, inspectable, testable function.

Two details that a naive implementation gets wrong
---------------------------------------------------
**The setpoint law is quadratic, not linear.** ``T = Kt * w * |w|`` and
``w = u * w_max``, so ``T = T_max * u * |u|``. Allocating in thrust space and
then writing ``u = T / T_max`` is wrong everywhere except 0 and +-1: at half
thrust it commands 0.5 where 0.707 is needed. The inverse is
``u = sign(T) * sqrt(|T| / T_max)``.

**Saturation must be uniform, not per-element.** If one thruster clips, clipping
only that one rotates the delivered wrench away from the commanded one -- ask
for hard surge, get surge plus a yaw you did not ask for. Scaling *all* thrusts
in a group by a common factor preserves direction and only loses magnitude.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .scn_parse import VehicleSpec

__all__ = ["AllocationResult", "ThrustAllocator", "DOF_NAMES", "DOF_INDEX"]

# Wrench rows, in body-frame NED order.
DOF_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")
DOF_INDEX = {name: i for i, name in enumerate(DOF_NAMES)}


@dataclass
class AllocationResult:
    """Outcome of one allocation, with enough detail to debug a bad command."""

    setpoints: np.ndarray  # (N,) in [-1, 1], ready to publish
    thrusts: np.ndarray  # (N,) signed thrust [N] actually commanded
    wrench_requested: np.ndarray  # (6,) body wrench asked for
    wrench_delivered: np.ndarray  # (6,) body wrench after saturation scaling
    saturation: float  # 1.0 = no clipping; <1 = scaled down by this factor


class ThrustAllocator:
    """Maps normalised wrench commands to thruster setpoints for a vehicle.

    Parameters
    ----------
    spec:
        Vehicle geometry, normally from :func:`sauvc_gym.scn_parse.parse_scenario`.
    action_dofs:
        Which DOFs the action vector controls. Defaults to the four the vehicle
        can hold authority over while staying passively stable in roll/pitch.
    group:
        If True (default), solve the horizontal and vertical thruster groups as
        two independent least-squares problems. This is not just an optimisation:
        it prevents the pseudo-inverse from "helpfully" using vertical thrusters
        to trim a yaw command, which is both physically silly and a nasty source
        of depth coupling.
    B:
        Optional 6xN allocation matrix override, e.g. one measured empirically by
        ``scripts/identify_allocation.py``. Defaults to the scene geometry.
    """

    HORIZONTAL_DOFS = ("surge", "sway", "yaw")
    VERTICAL_DOFS = ("heave", "roll", "pitch")

    def __init__(
        self,
        spec: VehicleSpec,
        action_dofs: tuple[str, ...] = ("surge", "sway", "heave", "yaw"),
        group: bool = True,
        B: np.ndarray | None = None,
    ) -> None:
        bad = set(action_dofs) - set(DOF_NAMES)
        if bad:
            raise ValueError(f"unknown DOFs {sorted(bad)}; valid: {DOF_NAMES}")

        self.spec = spec
        self.action_dofs = tuple(action_dofs)
        self.group = group
        self.B = spec.allocation_matrix() if B is None else np.asarray(B, dtype=float)

        if self.B.shape != (6, spec.n_thrusters):
            raise ValueError(f"B must be 6x{spec.n_thrusters}, got {self.B.shape}")

        self.t_max = spec.max_thrusts

        # Partition thrusters by which wrench subspace they mostly serve. We do
        # this by looking at the actual axis, not by trusting a naming scheme:
        # a thruster whose axis is mostly +-Z is a "vertical".
        z_align = np.abs(self.B[2, :]) / (np.linalg.norm(self.B[0:3, :], axis=0) + 1e-12)
        self._vertical = z_align > 0.5
        self._horizontal = ~self._vertical

        self._groups: list[tuple[np.ndarray, tuple[str, ...]]] = []
        if self.group:
            if self._horizontal.any():
                self._groups.append((self._horizontal, self.HORIZONTAL_DOFS))
            if self._vertical.any():
                self._groups.append((self._vertical, self.VERTICAL_DOFS))
        else:
            self._groups.append((np.ones(spec.n_thrusters, dtype=bool), DOF_NAMES))

        self.tau_max = self._compute_axis_limits()

    # ---------------------------------------------------------------- limits

    def _compute_axis_limits(self) -> np.ndarray:
        """Peak wrench available on each axis considered alone.

        Used purely to normalise the action space, so that ``a = 1`` means "as
        much surge as this vehicle has" rather than an arbitrary number of
        Newtons. For axis j the best case is every thruster pushing with its
        sign, i.e. ``sum_i |B[j, i]| * T_max_i``.
        """
        limits = np.abs(self.B) @ self.t_max
        # An axis with no authority would make normalisation blow up.
        return np.where(limits < 1e-9, 1.0, limits)

    # ------------------------------------------------------------ allocation

    def action_to_wrench(self, action: np.ndarray) -> np.ndarray:
        """Expand a normalised action over ``action_dofs`` into a 6-D wrench."""
        action = np.asarray(action, dtype=float).reshape(-1)
        if action.shape[0] != len(self.action_dofs):
            raise ValueError(
                f"action has {action.shape[0]} entries, expected "
                f"{len(self.action_dofs)} for {self.action_dofs}"
            )
        action = np.clip(action, -1.0, 1.0)

        wrench = np.zeros(6, dtype=float)
        for value, name in zip(action, self.action_dofs):
            j = DOF_INDEX[name]
            wrench[j] = value * self.tau_max[j]
        return wrench

    def allocate(self, action: np.ndarray) -> AllocationResult:
        """Full pipeline: normalised action -> thruster setpoints."""
        wrench_req = self.action_to_wrench(action)
        thrusts = np.zeros(self.spec.n_thrusters, dtype=float)
        worst_scale = 1.0

        for mask, dofs in self._groups:
            rows = [DOF_INDEX[d] for d in dofs]
            b_sub = self.B[np.ix_(rows, np.where(mask)[0])]
            tau_sub = wrench_req[rows]

            # Minimum-norm least squares: among all thrust vectors delivering
            # this wrench, take the one with least total effort. For the
            # horizontal group (3 equations, 4 thrusters) this resolves the
            # 1-D nullspace; for the vertical group it delivers heave while
            # holding roll/pitch moments at zero.
            t_sub, *_ = np.linalg.lstsq(b_sub, tau_sub, rcond=None)

            # Direction-preserving saturation, per group. The groups are
            # physically decoupled, so scaling them independently does not
            # distort either delivered wrench.
            limit = self.t_max[mask]
            over = np.abs(t_sub) / limit
            peak = float(over.max()) if over.size else 0.0
            if peak > 1.0:
                t_sub = t_sub / peak
                worst_scale = min(worst_scale, 1.0 / peak)

            thrusts[mask] = t_sub

        setpoints = np.array(
            [t.setpoint_from_thrust(thrusts[i]) for i, t in enumerate(self.spec.thrusters)],
            dtype=float,
        )

        return AllocationResult(
            setpoints=setpoints,
            thrusts=thrusts,
            wrench_requested=wrench_req,
            wrench_delivered=self.B @ thrusts,
            saturation=worst_scale,
        )

    def setpoints_to_wrench(self, setpoints: np.ndarray) -> np.ndarray:
        """Forward model: what wrench do these setpoints actually produce?

        The inverse of :meth:`allocate`, used by the tests to prove the loop
        closes and by the identification script to check measured vs declared.
        """
        setpoints = np.asarray(setpoints, dtype=float).reshape(-1)
        thrusts = np.array(
            [t.thrust_from_setpoint(u) for t, u in zip(self.spec.thrusters, setpoints)],
            dtype=float,
        )
        return self.B @ thrusts

    # ----------------------------------------------------------- diagnostics

    def describe(self) -> str:
        """Human-readable summary, printed by the verification script."""
        lines = [
            f"vehicle          : {self.spec.robot_name}",
            f"source           : {self.spec.source_file}",
            f"thrusters        : {self.spec.n_thrusters} {self.spec.names}",
            f"horizontal group : {[self.spec.names[i] for i in np.where(self._horizontal)[0]]}",
            f"vertical group   : {[self.spec.names[i] for i in np.where(self._vertical)[0]]}",
            f"action DOFs      : {self.action_dofs}",
            "",
            "per-thruster:",
        ]
        for i, t in enumerate(self.spec.thrusters):
            lines.append(
                f"  {t.name:<6} pos=({t.position[0]:+.3f},{t.position[1]:+.3f},"
                f"{t.position[2]:+.3f})  axis=({t.direction[0]:+.3f},"
                f"{t.direction[1]:+.3f},{t.direction[2]:+.3f})  "
                f"Tmax={t.max_thrust:5.1f}N  sign={t.setpoint_sign:+.0f}"
            )
        lines += ["", "axis authority (both directions, all thrusters at max):"]
        units = ["N", "N", "N", "Nm", "Nm", "Nm"]
        for j, name in enumerate(DOF_NAMES):
            lines.append(f"  {name:<6} {self.tau_max[j]:8.2f} {units[j]}")
        return "\n".join(lines)
