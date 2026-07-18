"""
Episode reset strategies.

The problem
-----------
Gymnasium assumes ``reset()`` puts the world back to a known state, cheaply and
exactly. Stonefish's stock ``stonefish_ros2`` simulator node offers no such
thing: it parses a scenario, starts free-running in real time, and exposes
sensors and actuator setpoints over topics. There is no teleport topic, no
pause, no rewind. So a reset has to be manufactured, and every way of doing it
trades something away.

The three ways, honestly
-------------------------
``soft``
    Fly the vehicle home with a PD controller, wait for it to settle, start the
    next episode. Fast-ish (5-20 s), no simulator changes, works today. But the
    initial state is only approximately the same each time, residual water
    motion carries across episodes, and any object the vehicle knocked over
    stays knocked over. Fine for station-keeping and control tasks; not fine for
    anything where props move.

``relaunch``
    Kill the simulator, start it again. Exactly clean, including props. Costs
    5-20 s of process startup and mesh loading per episode -- with a 1.17M-face
    CAD model, closer to the top of that range. Use it as ground truth to check
    that ``soft`` is not quietly corrupting your episodes.

``service``
    Call a reset service on a *custom* simulator node. This is the only option
    that is both fast and exact, and the only one that can also decouple the env
    from the wall clock -- but it requires writing that node. See
    ``docs/STEPPED_BACKEND.md``; the hooks exist in Stonefish >= 1.5, which added
    manual stepping of the simulation specifically for RL work.

Default is ``soft``, because it is the only one that runs against your current
workspace with no changes.
"""

from __future__ import annotations

import abc
import os
import signal
import subprocess
import time

import numpy as np

from .allocation import ThrustAllocator
from .ros_link import RosLink, wrap_pi

__all__ = ["ResetBackend", "SoftResetBackend", "RelaunchBackend", "ServiceResetBackend",
           "make_backend"]


class ResetBackend(abc.ABC):
    """Puts the simulated world back to a start state."""

    @abc.abstractmethod
    def reset(self, target_pose: np.ndarray, rng: np.random.Generator) -> bool:
        """Return the world to ``target_pose`` = (x, y, z, yaw). True on success."""

    def close(self) -> None:
        """Release anything the backend owns."""


class SoftResetBackend(ResetBackend):
    """Drives the vehicle home with a PD controller. No simulator changes needed.

    This is a controller, not a teleport, so it is subject to the same physics as
    the policy. That is a feature for realism and a nuisance for throughput: a
    reset costs real seconds and lands within a tolerance, not exactly.

    The gains are intentionally sedate. This runs thousands of times unattended
    and a reset controller that overshoots and oscillates will quietly poison
    every episode after it.
    """

    def __init__(
        self,
        link: RosLink,
        allocator: ThrustAllocator,
        pos_tol: float = 0.15,
        yaw_tol: float = 0.10,
        vel_tol: float = 0.05,
        timeout: float = 30.0,
        settle_time: float = 1.0,
        control_hz: float = 20.0,
        kp_pos: float = 0.6,
        kd_pos: float = 0.9,
        kp_yaw: float = 0.8,
        kd_yaw: float = 0.4,
    ) -> None:
        self.link = link
        self.allocator = allocator
        self.pos_tol = pos_tol
        self.yaw_tol = yaw_tol
        self.vel_tol = vel_tol
        self.timeout = timeout
        self.settle_time = settle_time
        self.dt = 1.0 / control_hz
        self.kp_pos, self.kd_pos = kp_pos, kd_pos
        self.kp_yaw, self.kd_yaw = kp_yaw, kd_yaw

    def _command(self, state, target_pose: np.ndarray) -> np.ndarray:
        """One PD step -> normalised action over (surge, sway, heave, yaw)."""
        from .ros_link import quat_to_matrix

        err_world = target_pose[:3] - state.position
        r_bw = quat_to_matrix(state.orientation)
        err_body = r_bw.T @ err_world  # world error -> body frame

        v = state.lin_vel_body
        surge = self.kp_pos * err_body[0] - self.kd_pos * v[0]
        sway = self.kp_pos * err_body[1] - self.kd_pos * v[1]
        heave = self.kp_pos * err_body[2] - self.kd_pos * v[2]

        _, _, yaw = state.rpy
        yaw_err = wrap_pi(float(target_pose[3]) - yaw)
        yaw_cmd = self.kp_yaw * yaw_err - self.kd_yaw * state.ang_vel_body[2]

        return np.clip([surge, sway, heave, yaw_cmd], -1.0, 1.0)

    def _at_target(self, state, target_pose: np.ndarray) -> bool:
        pos_ok = float(np.linalg.norm(target_pose[:3] - state.position)) < self.pos_tol
        yaw_ok = abs(wrap_pi(float(target_pose[3]) - state.rpy[2])) < self.yaw_tol
        vel_ok = float(np.linalg.norm(state.lin_vel_body)) < self.vel_tol
        return pos_ok and yaw_ok and vel_ok

    def reset(self, target_pose: np.ndarray, rng: np.random.Generator) -> bool:
        target_pose = np.asarray(target_pose, dtype=float)
        deadline = time.monotonic() + self.timeout
        settled_since: float | None = None

        while time.monotonic() < deadline:
            state = self.link.get_state()
            action = self._command(state, target_pose)
            self.link.send_setpoints(self.allocator.allocate(action).setpoints)

            if self._at_target(state, target_pose):
                settled_since = settled_since or time.monotonic()
                if time.monotonic() - settled_since >= self.settle_time:
                    self.link.stop()
                    time.sleep(0.1)
                    return True
            else:
                settled_since = None

            time.sleep(self.dt)

        # Timed out. Do not silently start an episode from wherever we drifted
        # to -- the caller decides whether that is acceptable.
        self.link.stop()
        return False


class RelaunchBackend(ResetBackend):
    """Restarts the whole simulator process. Exact, clean, and slow.

    Use for evaluation, for tasks with movable props (the flares, the drums),
    and periodically during training to confirm ``soft`` has not drifted.
    """

    def __init__(
        self,
        launch_cmd: list[str],
        link_factory,
        startup_timeout: float = 90.0,
        env: dict[str, str] | None = None,
    ) -> None:
        self.launch_cmd = launch_cmd
        self.link_factory = link_factory
        self.startup_timeout = startup_timeout
        self.env = {**os.environ, **(env or {})}
        self.proc: subprocess.Popen | None = None
        self.link: RosLink | None = None

    def _kill(self) -> None:
        if self.proc is None:
            return
        # ros2 launch spawns children; kill the group or Stonefish outlives us
        # and the next instance fights it for the setpoint topic.
        try:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGINT)
            self.proc.wait(timeout=10.0)
        except Exception:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
            except Exception:
                pass
        self.proc = None

    def reset(self, target_pose: np.ndarray, rng: np.random.Generator) -> bool:
        if self.link is not None:
            self.link.close()
            self.link = None
        self._kill()

        self.proc = subprocess.Popen(
            self.launch_cmd,
            env=self.env,
            preexec_fn=os.setsid,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.link = self.link_factory()
        return self.link.wait_for_data(timeout=self.startup_timeout)

    def close(self) -> None:
        if self.link is not None:
            self.link.close()
        self._kill()


class ServiceResetBackend(ResetBackend):
    """Calls a reset service on a custom simulator node.

    Requires the node described in ``docs/STEPPED_BACKEND.md``. This class is
    the client half only -- deliberately, because the server half depends on
    Stonefish API symbols this package cannot verify for your build, and a
    plausible-looking wrong guess is worse than an honest gap.
    """

    def __init__(self, link: RosLink, service_name: str = "/sauvc_sim/reset",
                 timeout: float = 10.0) -> None:
        from std_srvs.srv import Trigger  # noqa: F401 - fail loudly if absent

        self.link = link
        self.timeout = timeout
        self.service_name = service_name
        self._client = None

    def reset(self, target_pose: np.ndarray, rng: np.random.Generator) -> bool:
        raise NotImplementedError(
            "ServiceResetBackend needs the custom simulator node. See "
            "docs/STEPPED_BACKEND.md for what it must do and which Stonefish "
            "symbols to confirm before wiring it up."
        )


def make_backend(name: str, **kwargs) -> ResetBackend:
    """Factory keyed by the ``reset_mode`` config string."""
    backends = {
        "soft": SoftResetBackend,
        "relaunch": RelaunchBackend,
        "service": ServiceResetBackend,
    }
    if name not in backends:
        raise ValueError(f"unknown reset_mode {name!r}; valid: {sorted(backends)}")
    return backends[name](**kwargs)
