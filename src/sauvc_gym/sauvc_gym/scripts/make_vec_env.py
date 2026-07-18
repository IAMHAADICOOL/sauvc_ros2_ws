#!/usr/bin/env python3
"""
Build a parallel vectorised env, one simulator instance per ROS_DOMAIN_ID.

This is the thing that makes ROS-coupled training practical. A single instance
runs at real time, so ~36k steps/hour at 10 Hz. Eight instances give ~290k/hour,
which turns a five-day PPO run into an overnight one.

Each worker needs its own ROS graph or the workers will publish setpoints to
each other's vehicles -- which fails in a confusing, intermittent, entertaining
way. ROS_DOMAIN_ID isolates them. Valid ids are 0-101 with the default RMW
settings; this uses a base of 30 to stay clear of whatever else you have running.

You must launch the simulators yourself, one per domain, headless::

    for i in 0 1 2 3 4 5 6 7; do
      ROS_DOMAIN_ID=$((30+i)) ros2 launch sauvc_stonefish sauvc_qualification.launch.py \
        gui:=false &
    done

Drop the cameras from the scenario for control tasks. They are the dominant
per-step cost and a station-keeping policy never reads them.
"""

from __future__ import annotations

import os


def make_env_fn(rank: int, scn: str, env_id: str = "SauvcStationKeeping-v0",
                base_domain: int = 30, **env_kwargs):
    """Return a thunk that builds one env in its own ROS domain."""

    def _init():
        # Must be set before rclpy.init() in this process, hence inside the
        # thunk: SubprocVecEnv calls it after the fork.
        os.environ["ROS_DOMAIN_ID"] = str(base_domain + rank)
        import gymnasium as gym

        import sauvc_gym  # noqa: F401
        from sauvc_gym.wrappers import EpisodeStats

        env = gym.make(env_id, vehicle_scn=scn, **env_kwargs)
        return EpisodeStats(env)

    return _init


def make_vec_env(n_envs: int, scn: str, env_id: str = "SauvcStationKeeping-v0",
                 base_domain: int = 30, **env_kwargs):
    """SubprocVecEnv over ``n_envs`` simulator instances.

    SubprocVecEnv, not DummyVecEnv: these envs block on network I/O, so they
    must be in separate processes to overlap. DummyVecEnv would serialise them
    and you would get no speedup at all.
    """
    from stable_baselines3.common.vec_env import SubprocVecEnv

    if base_domain + n_envs > 101:
        raise ValueError("ROS_DOMAIN_ID would exceed 101; lower base_domain")

    return SubprocVecEnv(
        [make_env_fn(i, scn, env_id, base_domain, **env_kwargs)
         for i in range(n_envs)],
        start_method="spawn",
    )
