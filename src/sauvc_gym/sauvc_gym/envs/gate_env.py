"""
Qualification gate: pass through the gate, turn, come back.

The rule being modelled
-----------------------
Touch the wall on the marked starting line, swim ~10 m, pass completely through
the qualification gate (150 cm wide, spanning surface to bottom), U-turn, and
pass through it again -- without surfacing, touching the bottom or walls, or
touching any part of the gate.

Read this before you train it
------------------------------
This env hands the policy the gate's bearing and range. **The real vehicle does
not have that.** On hardware the gate comes from the forward camera, through a
detector you have not built yet, at a rate and reliability nothing here models.
A policy trained against this observation is therefore not deployable as-is. It
is still worth building, for two reasons:

* It validates the reward, the termination logic and the arena wiring against a
  task you can score, without also debugging a perception stack.
* It gives the upper bound. Whatever this policy achieves with perfect gate
  knowledge is the ceiling a camera-fed version will approach from below, and
  the gap between the two is exactly the value of your detector.

When you do close the loop, swap ``gate_observation_source`` to ``"camera"``
and feed the detector's output in. The honest intermediate step is
``"noisy"``, which corrupts the ground-truth bearing with the error statistics
you measure from your actual detector -- that is a far better proxy than perfect
knowledge, and cheap to do.
"""

from __future__ import annotations

import numpy as np

from ..ros_link import VehicleState, quat_to_matrix, wrap_pi
from .auv_base_env import AuvBaseEnv

__all__ = ["QualificationGateEnv"]


class QualificationGateEnv(AuvBaseEnv):
    """Transit the gate outbound, then inbound.

    Parameters
    ----------
    gate_position:
        Gate centre (x, y) in pool coordinates. Defaults to the qualification
        layout: starting line at the wall (x = -12.5), gate 10 m along.
    gate_width:
        Clear opening [m]. The vehicle must pass inside this, not through a post.
    gate_observation_source:
        ``"ground_truth"`` (default, not deployable), ``"noisy"`` (ground truth
        plus detector-like error) or ``"camera"`` (you supply it).
    bearing_noise_std, range_noise_std, dropout_prob:
        Only used when the source is ``"noisy"``. Set them from measurements of
        your real detector, not from taste.
    """

    def __init__(
        self,
        *args,
        gate_position: tuple[float, float] = (-2.5, 0.0),
        gate_width: float = 1.5,
        gate_observation_source: str = "ground_truth",
        bearing_noise_std: float = 0.05,
        range_noise_std: float = 0.3,
        dropout_prob: float = 0.1,
        w_progress: float = 1.0,
        w_align: float = 0.3,
        w_depth: float = 0.2,
        w_effort: float = 0.02,
        pass_bonus: float = 20.0,
        crash_penalty: float = 20.0,
        **kwargs,
    ) -> None:
        if gate_observation_source not in ("ground_truth", "noisy", "camera"):
            raise ValueError(
                "gate_observation_source must be 'ground_truth', 'noisy' or 'camera'"
            )
        self.gate_position = np.asarray(gate_position, dtype=float)
        self.gate_width = gate_width
        self.gate_observation_source = gate_observation_source
        self.bearing_noise_std = bearing_noise_std
        self.range_noise_std = range_noise_std
        self.dropout_prob = dropout_prob
        self.w_progress = w_progress
        self.w_align = w_align
        self.w_depth = w_depth
        self.w_effort = w_effort
        self.pass_bonus = pass_bonus
        self.crash_penalty = crash_penalty

        self._legs_done = 0
        self._prev_side = 0
        self._prev_dist = 0.0
        self._last_gate_obs = np.zeros(4, dtype=np.float32)

        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------ observation

    def _task_observation_dim(self) -> int:
        # (sin, cos) of gate bearing, normalised range, legs completed
        return 4

    def _gate_relative(self, state: VehicleState) -> tuple[float, float]:
        """True bearing [rad] and range [m] to the gate, in the body frame."""
        delta = np.array(
            [
                self.gate_position[0] - state.position[0],
                self.gate_position[1] - state.position[1],
                0.0,
            ]
        )
        body = quat_to_matrix(state.orientation).T @ delta
        return float(np.arctan2(body[1], body[0])), float(np.linalg.norm(body[:2]))

    def _task_observation(self, state: VehicleState) -> np.ndarray:
        bearing, rng = self._gate_relative(state)

        if self.gate_observation_source == "noisy":
            if self.np_random.random() < self.dropout_prob:
                # A dropout must not look like "gate dead ahead at zero range".
                # Reuse the last good measurement, which is what any sane
                # tracker would do, and what your mission code will do too.
                return self._last_gate_obs
            bearing += self.np_random.normal(0.0, self.bearing_noise_std)
            rng += self.np_random.normal(0.0, self.range_noise_std)
        elif self.gate_observation_source == "camera":
            raise NotImplementedError(
                "Wire your detector in here: subscribe to its output and return "
                "(sin b, cos b, range/25, legs/2). Until then use 'noisy'."
            )

        obs = np.array(
            [np.sin(bearing), np.cos(bearing), rng / 25.0, self._legs_done / 2.0],
            dtype=np.float32,
        )
        self._last_gate_obs = obs
        return obs

    # ---------------------------------------------------------------- episode

    def _sample_goal(self, rng: np.random.Generator) -> np.ndarray:
        goal = self.start_pose.copy()
        goal[2] = rng.uniform(0.5, 1.0)  # transit depth
        goal[3] = 0.0
        return goal

    def reset(self, *, seed=None, options=None):
        self._legs_done = 0
        obs, info = super().reset(seed=seed, options=options)
        state = self.link.get_state()
        self._prev_side = int(np.sign(state.position[0] - self.gate_position[0]))
        self._prev_dist = float(
            np.linalg.norm(state.position[:2] - self.gate_position)
        )
        info["legs_done"] = 0
        return obs, info

    def _crossed_gate(self, state: VehicleState) -> bool:
        """Did we just pass through the opening, rather than beside a post?

        Crossing is detected on the sign change of the along-pool offset. The
        lateral check is what separates "went through the gate" from "went
        around it", and it is the entire difference between a legal run and a
        zero.
        """
        side = int(np.sign(state.position[0] - self.gate_position[0]))
        if side == 0 or side == self._prev_side:
            return False
        self._prev_side = side
        within = abs(state.position[1] - self.gate_position[1]) < self.gate_width / 2
        return bool(within)

    def _reward(self, state: VehicleState, action: np.ndarray) -> float:
        dist = float(np.linalg.norm(state.position[:2] - self.gate_position))

        # Progress: reward closing the range, per step. Potential-based, so it
        # sums to the total distance closed and cannot be farmed by orbiting.
        r_progress = self.w_progress * (self._prev_dist - dist)
        self._prev_dist = dist

        bearing, _ = self._gate_relative(state)
        r_align = self.w_align * np.exp(-((bearing / 0.5) ** 2))
        r_depth = self.w_depth * np.exp(-(((state.depth - self._goal[2]) / 0.2) ** 2))
        r_effort = -self.w_effort * float(np.sum(np.square(action)))

        reward = r_progress + r_align + r_depth + r_effort

        if self._crossed_gate(state):
            self._legs_done += 1
            reward += self.pass_bonus
            # After the outbound pass, the objective inverts: the gate stops
            # being a thing to approach and becomes a thing to come back through.
            self._prev_dist = dist

        terminated, _ = self._safety_terminated(state)
        if terminated:
            reward -= self.crash_penalty
        return float(reward)

    def _task_terminated(self, state: VehicleState) -> tuple[bool, str]:
        if self._legs_done >= 2:
            return True, "gate_run_complete"
        return False, ""
