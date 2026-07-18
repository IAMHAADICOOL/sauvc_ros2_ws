"""
Base Gymnasium environment wrapping the SAUVC Stonefish arena.

The load-bearing design decision
---------------------------------
**Reward may read ground truth. Observation may not.**

The simulator publishes ``/sauvc_auv/odometry``, which is perfect pose. It is
tempting to feed that straight to the policy. Doing so trains an agent that
cannot exist: the real vehicle has an IMU, a pressure sensor, and two cameras,
and its best pose estimate is whatever the robot_localization EKF produces --
drifting in x/y, decent in depth and heading. A policy conditioned on perfect
x/y will fall over the moment it meets an EKF.

So this env splits the two:

* ``_observation()`` may use only quantities the hardware can actually produce:
  depth (Bar30), attitude and rates (HFI-A9), body velocities (EKF / optical
  flow), and the previous action. This is the *deployable* interface.
* ``_reward()`` and ``_terminated()`` may use anything, including ground truth.
  They exist only during training and are never shipped.

This is standard privileged-critic practice, and it is the difference between a
policy that transfers and a demo.

Sample throughput -- read this before you start a training run
---------------------------------------------------------------
This env is coupled to a free-running, wall-clock simulator. At the default
10 Hz, one instance produces ~36,000 steps per hour, and no amount of
optimisation in this file changes that -- the bound is the simulator's clock,
not the Python.

PPO on a task like station-keeping wants somewhere in the 1-5M step range. On a
single instance that is roughly **30 to 140 hours**. Options, in increasing
order of effort:

1. Run N headless instances on separate ``ROS_DOMAIN_ID``s with SB3's
   ``SubprocVecEnv`` (see ``scripts/make_vec_env.py``). 8 workers -> ~290k
   steps/h -> a few hours to overnight. This is the practical answer, and it is
   why your existing headless build matters.
2. Drop cameras from the scenario for control tasks. They are the dominant cost
   and a station-keeping policy does not read them.
3. Build the stepped backend (``docs/STEPPED_BACKEND.md``) and break the
   real-time coupling entirely.

Worth knowing: the Stonefish authors hit exactly this wall. Their ICRA 2025
paper reports that connecting Gym to Stonefish over a ROS interface was measured
to slow training down, and their answer was to bypass ROS with direct Python
bindings plus console mode. This package takes the ROS route because it reuses
your working arena, bridge, and sensor wiring unchanged -- but go in knowing it
is the slow road, and that option 1 is what makes it tolerable.
"""

from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ..allocation import ThrustAllocator
from ..ros_link import RosLink, VehicleState, wrap_pi
from ..scn_parse import parse_scenario

__all__ = ["AuvBaseEnv", "PoolGeometry"]


class PoolGeometry:
    """SAUVC arena bounds, including the V-shaped floor.

    The floor is not flat and pretending otherwise breaks bottom-contact
    termination near the walls, where there is 40 cm less water than at the
    centre. Profile: ``d(x) = 1.6 - 0.032 * |x|``, so 1.6 m at the centreline
    and 1.2 m at the end walls.
    """

    LENGTH = 25.0  # x, -12.5 .. +12.5
    WIDTH = 16.0  # y, -8.0 .. +8.0
    DEPTH_CENTRE = 1.6
    FLOOR_SLOPE = 0.032

    def __init__(self, use_floor_profile: bool = True, flat_depth: float = 2.0) -> None:
        self.use_floor_profile = use_floor_profile
        self.flat_depth = flat_depth

    def floor_depth(self, x: float) -> float:
        """Depth of the pool floor [m] at along-pool position ``x``."""
        if not self.use_floor_profile:
            return self.flat_depth
        return self.DEPTH_CENTRE - self.FLOOR_SLOPE * abs(float(x))

    def contains(self, position: np.ndarray, margin: float = 0.3) -> bool:
        x, y, z = float(position[0]), float(position[1]), float(position[2])
        return (
            abs(x) < self.LENGTH / 2 - margin
            and abs(y) < self.WIDTH / 2 - margin
            and z > margin
            and z < self.floor_depth(x) - margin
        )


class AuvBaseEnv(gym.Env):
    """Common machinery: spaces, stepping, safety, episode bookkeeping.

    Subclasses supply the task by overriding :meth:`_task_observation`,
    :meth:`_reward`, :meth:`_task_terminated` and :meth:`_sample_goal`.

    Parameters
    ----------
    vehicle_scn:
        Path to the *vehicle* scenario, for thruster geometry. Not the arena file.
    robot_name:
        Must match the scenario's robot name and topic namespace.
    control_hz:
        Action rate. 10 Hz matches a sane outer-loop rate on the Jetson and is
        slow enough that the sim keeps up. Do not raise it without checking the
        real-time factor that ``info["rtf"]`` reports.
    action_dofs:
        Which axes the policy commands. Default is the four this hull can hold
        authority over while staying passively stable in roll/pitch.
    reset_mode:
        ``"soft"``, ``"relaunch"`` or ``"service"``. See :mod:`sauvc_gym.backends`.
    """

    metadata = {"render_modes": [], "render_fps": 10}

    # Observation scaling. Not cosmetic: PPO's default network assumes roughly
    # unit-variance inputs, and raw metres-per-second alongside raw radians per
    # second trains badly. These are rough physical maxima for this vehicle.
    OBS_SCALE_VEL = 2.0  # m/s
    OBS_SCALE_RATE = 3.0  # rad/s
    OBS_SCALE_DEPTH = 2.0  # m

    def __init__(
        self,
        vehicle_scn: str,
        robot_name: str = "sauvc_auv",
        control_hz: float = 10.0,
        action_dofs: tuple[str, ...] = ("surge", "sway", "heave", "yaw"),
        max_episode_steps: int = 600,
        reset_mode: str = "soft",
        start_pose: tuple[float, float, float, float] = (-11.4, 0.0, 1.0, 0.0),
        use_floor_profile: bool = True,
        odom_twist_frame: str = "world",
        step_timeout: float = 5.0,
        allocation_matrix: np.ndarray | None = None,
        link: RosLink | None = None,
        domain_id: int | None = None,
        **backend_kwargs: Any,
    ) -> None:
        super().__init__()

        self.spec_vehicle = parse_scenario(vehicle_scn, robot_name=robot_name)
        self.allocator = ThrustAllocator(
            self.spec_vehicle, action_dofs=action_dofs, B=allocation_matrix
        )
        self.pool = PoolGeometry(use_floor_profile=use_floor_profile)

        self.control_hz = float(control_hz)
        self.dt = 1.0 / self.control_hz
        self.max_episode_steps = int(max_episode_steps)
        self.start_pose = np.asarray(start_pose, dtype=float)
        self.step_timeout = step_timeout

        self.link = link or RosLink(
            robot_name=robot_name,
            setpoint_topic=self.spec_vehicle.setpoint_topic,
            n_thrusters=self.spec_vehicle.n_thrusters,
            odom_twist_frame=odom_twist_frame,
            domain_id=domain_id,
        )
        if not self.link.wait_for_data(timeout=60.0):
            raise RuntimeError(
                f"No odometry on {self.link.odom_topic} after 60 s. Is the "
                f"simulator running, and does robot_name={robot_name!r} match "
                f"the scenario? Check: ros2 topic hz {self.link.odom_topic}"
            )

        from ..backends import SoftResetBackend, make_backend

        if reset_mode == "soft":
            self.backend = SoftResetBackend(
                link=self.link, allocator=self.allocator, **backend_kwargs
            )
        else:
            self.backend = make_backend(reset_mode, link=self.link, **backend_kwargs)

        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(len(action_dofs),), dtype=np.float32
        )
        obs_dim = self._observation_dim()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        self._prev_action = np.zeros(len(action_dofs), dtype=np.float32)
        self._step_count = 0
        self._goal: np.ndarray = self.start_pose.copy()
        self._last_state: VehicleState = self.link.get_state()
        self._last_wall = time.monotonic()
        self._rtf = 1.0

    # ------------------------------------------------------- task interface

    def _task_observation(self, state: VehicleState) -> np.ndarray:
        """Task-specific observation terms. Deployable quantities only."""
        return np.zeros(0, dtype=np.float32)

    def _task_observation_dim(self) -> int:
        return 0

    def _reward(self, state: VehicleState, action: np.ndarray) -> float:
        raise NotImplementedError

    def _task_terminated(self, state: VehicleState) -> tuple[bool, str]:
        return False, ""

    def _sample_goal(self, rng: np.random.Generator) -> np.ndarray:
        return self.start_pose.copy()

    # ------------------------------------------------------------ observation

    def _observation_dim(self) -> int:
        return 11 + len(self.allocator.action_dofs) + self._task_observation_dim()

    def _base_observation(self, state: VehicleState) -> np.ndarray:
        """The eleven quantities the real vehicle can also produce.

        depth error (1)   -- Bar30, minus the commanded depth
        body velocity (3) -- EKF, fed by optical flow and IMU
        roll, pitch (2)   -- HFI-A9 fused orientation
        yaw error (2)     -- as sin/cos, so the policy never sees a wrap
        body rates (3)    -- HFI-A9 gyro

        Yaw error is encoded as a (sin, cos) pair rather than an angle on
        purpose. A raw wrapped angle has a discontinuity at +-pi that a network
        has to waste capacity learning around, and which produces a genuinely
        wrong gradient right where heading control matters most.
        """
        roll, pitch, yaw = state.rpy
        depth_err = (state.depth - self._goal[2]) / self.OBS_SCALE_DEPTH
        yaw_err = wrap_pi(float(self._goal[3]) - yaw)

        return np.array(
            [
                depth_err,
                *(state.lin_vel_body / self.OBS_SCALE_VEL),
                roll,
                pitch,
                np.sin(yaw_err),
                np.cos(yaw_err),
                *(state.ang_vel_body / self.OBS_SCALE_RATE),
            ],
            dtype=np.float32,
        )

    def _observation(self, state: VehicleState) -> np.ndarray:
        return np.concatenate(
            [
                self._base_observation(state),
                self._prev_action,
                self._task_observation(state),
            ]
        ).astype(np.float32)

    # -------------------------------------------------------------- safety

    def _safety_terminated(self, state: VehicleState) -> tuple[bool, str]:
        """Conditions that end an episode regardless of task.

        These mirror the SAUVC rules where they can: surfacing and touching the
        bottom are both disqualifying in the qualification run, so an agent
        should never be rewarded for a trajectory that does either.
        """
        x, y, z = state.position
        roll, pitch, _ = state.rpy

        if not np.all(np.isfinite(state.position)):
            return True, "diverged"
        if z < 0.10:
            return True, "surfaced"
        if z > self.pool.floor_depth(x) - 0.05:
            return True, "bottom_contact"
        if abs(x) > self.pool.LENGTH / 2 - 0.2 or abs(y) > self.pool.WIDTH / 2 - 0.2:
            return True, "wall_contact"
        if abs(roll) > np.deg2rad(60) or abs(pitch) > np.deg2rad(60):
            return True, "capsized"
        return False, ""

    # ------------------------------------------------------------- gym API

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)

        self.link.stop()
        self._goal = self._sample_goal(self.np_random)

        start = (options or {}).get("start_pose", self.start_pose)
        ok = self.backend.reset(np.asarray(start, dtype=float), self.np_random)
        if not ok:
            # Surfacing this rather than swallowing it: a reset that quietly
            # fails produces episodes starting from arbitrary states, and the
            # resulting learning curve looks like a hyperparameter problem.
            import warnings

            warnings.warn(
                "reset did not reach the start pose within its timeout; this "
                "episode begins from wherever the vehicle drifted to. Repeated "
                "occurrences mean the reset gains or timeout need attention.",
                RuntimeWarning,
                stacklevel=2,
            )

        self._prev_action[:] = 0.0
        self._step_count = 0
        self._last_state = self.link.get_state()
        self._last_wall = time.monotonic()

        obs = self._observation(self._last_state)
        return obs, {"reset_ok": ok, "goal": self._goal.copy()}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)

        result = self.allocator.allocate(action)
        self.link.send_setpoints(result.setpoints)

        # Synchronise on the simulator's own output, not on our clock. If
        # Stonefish stalls we wait; if it is faster than real time we still only
        # take one action per published state, which keeps the transition
        # dynamics consistent.
        before = self._last_state
        tick_deadline = self._last_wall + self.dt
        state = self.link.wait_for_new_state(before.seq, timeout=self.step_timeout)

        # Hold the rest of the control period. Odometry is published far faster
        # than we act, so without this the env would run as fast as the sensor
        # rate and the policy would be trained at a dt it will never see.
        remaining = tick_deadline - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
            state = self.link.get_state()

        now = time.monotonic()
        wall_dt = now - self._last_wall
        sim_dt = state.stamp - before.stamp
        if wall_dt > 1e-6 and sim_dt > 0:
            self._rtf = 0.9 * self._rtf + 0.1 * (sim_dt / wall_dt)
        self._last_wall = now
        self._last_state = state
        self._step_count += 1

        reward = self._reward(state, action)

        terminated, reason = self._safety_terminated(state)
        if not terminated:
            terminated, reason = self._task_terminated(state)
        truncated = self._step_count >= self.max_episode_steps

        if terminated:
            self.link.stop()

        self._prev_action = action.copy()
        obs = self._observation(state)

        info = {
            # Ground truth: legitimate for logging and analysis, never for the
            # policy. Kept here rather than in the observation on purpose.
            "position": state.position.copy(),
            "rpy": np.array(state.rpy),
            "lin_vel_body": state.lin_vel_body.copy(),
            "goal": self._goal.copy(),
            "setpoints": result.setpoints.copy(),
            "saturation": result.saturation,
            "wrench_requested": result.wrench_requested.copy(),
            "wrench_delivered": result.wrench_delivered.copy(),
            "rtf": self._rtf,
            "sim_dt": sim_dt,
            "termination_reason": reason,
        }
        if terminated:
            info["terminal_reason"] = reason

        return obs, float(reward), bool(terminated), bool(truncated), info

    def close(self) -> None:
        try:
            self.link.stop()
        finally:
            self.backend.close()
            self.link.close()
