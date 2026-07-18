#!/usr/bin/env python3
"""
A PPO baseline. Deliberately unambitious.

    python3 -m sauvc_gym.scripts.train_sb3 --scn my_auv.scn --envs 8 --steps 2000000

The hyperparameters are SB3 defaults with three changes, each for a reason:

  n_steps=512      -- at 10 Hz that is ~51 s of vehicle time per rollout, long
                      enough to contain the settling transient we care about.
  gamma=0.995      -- effective horizon ~200 steps = 20 s. Station keeping is a
                      long-horizon regulation problem; the 0.99 default
                      discounts away the drift we are trying to punish.
  VecNormalize     -- observations here mix m/s, rad and rad/s. The base env
                      pre-scales them roughly, but running normalisation earns
                      its keep and costs nothing.

Everything else is stock, on purpose: if this does not learn depth hold, the
problem is the environment, not the entropy coefficient. Fix the env first.

Save VecNormalize's statistics alongside the model. A policy loaded without them
sees differently-scaled inputs and behaves like an untrained network -- a very
common and very confusing way to lose a working result.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scn", required=True)
    p.add_argument("--env-id", default="SauvcStationKeeping-v0")
    p.add_argument("--envs", type=int, default=1)
    p.add_argument("--steps", type=int, default=1_000_000)
    p.add_argument("--base-domain", type=int, default=30)
    p.add_argument("--out", default="runs/ppo_station")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.vec_env import VecNormalize
    except ImportError:
        print("pip install stable-baselines3", file=sys.stderr)
        return 1

    from sauvc_gym.scripts.make_vec_env import make_vec_env

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    venv = make_vec_env(args.envs, args.scn, args.env_id, args.base_domain)
    venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

    hours = args.steps / (args.envs * 36_000)
    print(f"{args.steps:,} steps across {args.envs} real-time instances "
          f"=> ~{hours:.1f} h wall clock. Ctrl-C now if that is not what you meant.")

    model = PPO("MlpPolicy", venv, n_steps=512, gamma=0.995, seed=args.seed,
                verbose=1, tensorboard_log=str(out / "tb"))
    model.learn(
        total_timesteps=args.steps,
        callback=CheckpointCallback(save_freq=max(20_000 // args.envs, 1),
                                    save_path=str(out / "ckpt"),
                                    name_prefix="ppo"),
    )

    model.save(out / "final")
    venv.save(str(out / "vecnormalize.pkl"))  # without this the model is useless
    venv.close()
    print(f"saved {out/'final.zip'} and {out/'vecnormalize.pkl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
