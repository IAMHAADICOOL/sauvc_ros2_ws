"""
Station keeping: hold a depth and a heading against disturbance.

Why this task first
-------------------
It is the smallest task that exercises every part of the stack -- allocation,
observation, reward, reset, termination -- while having a known-good baseline
you already own: ``sauvc_motion_demo``'s depth PID. If a learned policy cannot
beat, or at least match, that PID on depth hold, the problem is in this
environment, not in the algorithm. Do not move to the gate task until this one
works, or you will be debugging two things at once.

It is also the task whose reward is honest. Depth and heading are both directly
observable on the real vehicle (Bar30, HFI-A9), so a policy trained here has no
privileged-information gap to fall through on deployment.
"""

from __future__ import annotations

import numpy as np

from ..ros_link import VehicleState, wrap_pi
from .auv_base_env import AuvBaseEnv

__all__ = ["StationKeepingEnv"]


class StationKeepingEnv(AuvBaseEnv):
    """Hold a commanded depth and heading; stay still otherwise.

    Parameters
    ----------
    depth_range:
        Depth is resampled each episode from this range, so the policy learns
        depth *control* rather than one memorised setpoint. Kept clear of both
        the surface and the shallowest floor (1.2 m at the end walls).
    yaw_range:
        Heading offset resampled per episode, in radians.
    randomise_goal:
        Set False to pin the goal, for A/B against the PID baseline.
    w_*:
        Reward weights. See :meth:`_reward` for what each buys you.
    """

    def __init__(
        self,
        *args,
        depth_range: tuple[float, float] = (0.5, 1.1),
        yaw_range: tuple[float, float] = (-np.pi, np.pi),
        randomise_goal: bool = True,
        w_depth: float = 1.0,
        w_yaw: float = 0.5,
        w_drift: float = 0.2,
        w_effort: float = 0.02,
        w_rate: float = 0.05,
        w_attitude: float = 0.1,
        crash_penalty: float = 10.0,
        **kwargs,
    ) -> None:
        self.depth_range = depth_range
        self.yaw_range = yaw_range
        self.randomise_goal = randomise_goal
        self.w_depth = w_depth
        self.w_yaw = w_yaw
        self.w_drift = w_drift
        self.w_effort = w_effort
        self.w_rate = w_rate
        self.w_attitude = w_attitude
        self.crash_penalty = crash_penalty
        super().__init__(*args, **kwargs)

    def _sample_goal(self, rng: np.random.Generator) -> np.ndarray:
        goal = self.start_pose.copy()
        if self.randomise_goal:
            goal[2] = rng.uniform(*self.depth_range)
            goal[3] = wrap_pi(float(rng.uniform(*self.yaw_range)))
        return goal

    def _reward(self, state: VehicleState, action: np.ndarray) -> float:
        """Dense shaped reward.

        Shape note: errors enter as ``exp(-(e/sigma)^2)`` rather than as ``-|e|``.
        A bare negative-distance reward is unbounded below, so early on the agent
        is dominated by how badly it is doing rather than by which action helped,
        and it learns to minimise variance by sitting still. A bounded kernel
        gives a clear gradient near the goal and saturates far away, which is the
        behaviour you want.

        The effort and action-rate terms are not decoration. Without them a
        policy converges on bang-bang thrusting -- fine in simulation, and on
        real T200s a way to cook ESCs and drain the pack. ``w_rate`` in
        particular is what stops the chattering that never survives contact with
        a real thruster's rise time.
        """
        depth_err = state.depth - self._goal[2]
        yaw_err = wrap_pi(float(self._goal[3]) - state.rpy[2])
        roll, pitch, _ = state.rpy

        r_depth = self.w_depth * np.exp(-((depth_err / 0.15) ** 2))
        r_yaw = self.w_yaw * np.exp(-((yaw_err / 0.35) ** 2))

        # Lateral drift, ground truth. Legitimate here (reward, not observation)
        # and necessary: nothing else penalises slowly sliding across the pool.
        drift = float(np.linalg.norm(state.position[:2] - self._goal[:2]))
        r_drift = -self.w_drift * np.tanh(drift / 2.0)

        r_effort = -self.w_effort * float(np.sum(np.square(action)))
        r_rate = -self.w_rate * float(np.sum(np.square(action - self._prev_action)))
        r_attitude = -self.w_attitude * (roll**2 + pitch**2)

        reward = r_depth + r_yaw + r_drift + r_effort + r_rate + r_attitude

        terminated, _ = self._safety_terminated(state)
        if terminated:
            reward -= self.crash_penalty
        return float(reward)


class DepthHoldEnv(StationKeepingEnv):
    """Depth only: a one-dimensional action space, for sanity checks.

    Useful as the very first thing to run. If a policy cannot learn to hold
    depth with a single heave command, nothing downstream is worth debugging.
    Compare directly against ``sauvc_motion_demo``'s PID.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("action_dofs", ("heave",))
        kwargs.setdefault("w_yaw", 0.0)
        kwargs.setdefault("w_drift", 0.0)
        super().__init__(*args, **kwargs)
