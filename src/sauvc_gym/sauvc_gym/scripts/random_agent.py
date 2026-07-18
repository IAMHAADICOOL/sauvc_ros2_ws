#!/usr/bin/env python3
"""
Drive the env with random actions. The smoke test before any training.

    python3 -m sauvc_gym.scripts.random_agent --scn my_auv.scn --episodes 2

What to watch:
  rtf     -- real-time factor. Below ~0.9 the simulator is not keeping up and
             your dt is a lie. Drop cameras or lower control_hz.
  reason  -- how episodes end. All "bottom_contact" on a random policy is
             normal; all "capsized" in the first second is a thruster sign bug,
             so stop and run identify_allocation.py.
  sat     -- fraction of steps with a saturated thruster group.
"""

from __future__ import annotations

import argparse
import sys

import gymnasium as gym
import numpy as np

import sauvc_gym  # noqa: F401  (registers the env ids)
from sauvc_gym.wrappers import EpisodeStats


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scn", required=True)
    p.add_argument("--env-id", default="SauvcStationKeeping-v0")
    p.add_argument("--robot", default="sauvc_auv")
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--zero", action="store_true",
                   help="command zero instead of random: checks buoyancy trim")
    args = p.parse_args(argv)

    env = gym.make(args.env_id, vehicle_scn=args.scn, robot_name=args.robot,
                   max_episode_steps=args.steps)
    env = EpisodeStats(env)

    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            print(f"\nepisode {ep}: obs dim {obs.shape[0]}, "
                  f"goal depth {info['goal'][2]:.2f} m, "
                  f"yaw {np.rad2deg(info['goal'][3]):+.0f} deg, "
                  f"reset_ok={info['reset_ok']}")

            total = 0.0
            while True:
                action = (np.zeros(env.action_space.shape, dtype=np.float32)
                          if args.zero else env.action_space.sample())
                obs, reward, terminated, truncated, info = env.step(action)
                total += reward
                if terminated or truncated:
                    st = info.get("episode_stats", {})
                    print(f"  ended after {st.get('length')} steps: "
                          f"{st.get('reason')}")
                    print(f"  return {total:8.2f} | depth err mean "
                          f"{st.get('depth_err_mean', float('nan')):.3f} m "
                          f"max {st.get('depth_err_max', float('nan')):.3f} m")
                    print(f"  saturated {st.get('saturated_frac', 0)*100:5.1f}% "
                          f"| effort {st.get('effort_mean', 0):.3f} "
                          f"| rtf {st.get('rtf_mean', 0):.3f}")
                    if st.get("rtf_mean", 1.0) < 0.9:
                        print("  ! real-time factor below 0.9 -- the sim is "
                              "lagging. See README 'Throughput'.")
                    break
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
