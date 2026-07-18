"""
Wrappers that exist for sim-to-real reasons, not convenience.

Gymnasium already ships ``FrameStackObservation``, ``NormalizeObservation``,
``TimeLimit`` and friends -- use those. What follows is only the things specific
to putting a learned policy on a real AUV.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np

__all__ = ["ActionSlewLimit", "SafetyShield", "EpisodeStats"]


class ActionSlewLimit(gym.ActionWrapper):
    """Bounds how fast the commanded wrench may change between steps.

    A T200 does not step from -1 to +1 in 100 ms; it has rotor inertia and the
    ESC has a ramp. Stonefish's thruster model can be configured with rotor
    dynamics, but if yours is zero-order then the simulated thruster *will*
    step instantly, and a policy trained against it learns bang-bang control
    that the real thruster cannot follow. The policy then meets hardware, its
    commands are low-passed by physics it never saw, and it fails.

    Constraining it here is the cheap fix. The better fix is a first-order rotor
    model in the scene file -- plant parameters belong in the plant. Treat this
    wrapper as the stopgap it is.

    Parameters
    ----------
    max_delta:
        Largest change in any action component per step, in normalised units.
        0.2 at 10 Hz means a full-scale swing takes ~1 s.
    """

    def __init__(self, env: gym.Env, max_delta: float = 0.2) -> None:
        super().__init__(env)
        if max_delta <= 0:
            raise ValueError("max_delta must be positive")
        self.max_delta = float(max_delta)
        self._prev = np.zeros(env.action_space.shape, dtype=np.float32)

    def action(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        delta = np.clip(action - self._prev, -self.max_delta, self.max_delta)
        self._prev = np.clip(self._prev + delta, -1.0, 1.0)
        return self._prev.copy()

    def reset(self, **kwargs):
        self._prev = np.zeros(self.env.action_space.shape, dtype=np.float32)
        return self.env.reset(**kwargs)


class SafetyShield(gym.Wrapper):
    """Overrides the policy when it commands something dangerous.

    During training this mostly gets in the way -- the agent needs to experience
    the pool floor to learn to avoid it. Its real use is on hardware and during
    evaluation runs: a learned policy has no guarantee it will not drive into
    the bottom at full heave, and unlike in simulation you cannot press reset on
    a flooded hull.

    Off by default. Turn it on for anything touching water.

    Parameters
    ----------
    depth_limits:
        (min, max) depth [m]. Heave is overridden to push away from a violated
        bound. Note the max should respect the V-shaped floor: the pool is only
        1.2 m deep at the end walls.
    """

    def __init__(
        self,
        env: gym.Env,
        depth_limits: tuple[float, float] = (0.25, 1.1),
        max_tilt_deg: float = 35.0,
    ) -> None:
        super().__init__(env)
        self.depth_limits = depth_limits
        self.max_tilt = np.deg2rad(max_tilt_deg)
        self.interventions = 0

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).copy()
        state = self.env.unwrapped.link.get_state()
        dofs = self.env.unwrapped.allocator.action_dofs

        if "heave" in dofs:
            i = dofs.index("heave")
            lo, hi = self.depth_limits
            if state.depth < lo:
                action[i] = max(action[i], 0.3)  # +heave is down, in NED
                self.interventions += 1
            elif state.depth > hi:
                action[i] = min(action[i], -0.3)
                self.interventions += 1

        roll, pitch, _ = state.rpy
        if abs(roll) > self.max_tilt or abs(pitch) > self.max_tilt:
            action[:] = 0.0
            self.interventions += 1

        obs, reward, terminated, truncated, info = self.env.step(action)
        info["shield_interventions"] = self.interventions
        return obs, reward, terminated, truncated, info


class EpisodeStats(gym.Wrapper):
    """Accumulates the per-episode numbers worth looking at.

    Mean absolute depth error and thruster saturation fraction tell you more
    about whether a policy is deployable than the return does. A policy with a
    great return that sits at 100% saturation is not a controller, it is a
    latch.
    """

    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        self._reset_stats()

    def _reset_stats(self) -> None:
        self._depth_err: list[float] = []
        self._sat: list[float] = []
        self._effort: list[float] = []
        self._rtf: list[float] = []
        self._return = 0.0
        self._len = 0

    def reset(self, **kwargs):
        self._reset_stats()
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        base = self.env.unwrapped

        self._return += reward
        self._len += 1
        self._depth_err.append(abs(info["position"][2] - info["goal"][2]))
        self._sat.append(1.0 if info["saturation"] < 0.999 else 0.0)
        self._effort.append(float(np.mean(np.abs(info["setpoints"]))))
        self._rtf.append(info["rtf"])
        del base

        if terminated or truncated:
            info["episode_stats"] = {
                "return": self._return,
                "length": self._len,
                "depth_err_mean": float(np.mean(self._depth_err)),
                "depth_err_max": float(np.max(self._depth_err)),
                "saturated_frac": float(np.mean(self._sat)),
                "effort_mean": float(np.mean(self._effort)),
                "rtf_mean": float(np.mean(self._rtf)),
                "reason": info.get("terminal_reason", "truncated"),
            }
        return obs, reward, terminated, truncated, info
