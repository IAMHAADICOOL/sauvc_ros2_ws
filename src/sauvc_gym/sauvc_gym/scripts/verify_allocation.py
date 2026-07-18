#!/usr/bin/env python3
"""
Check the allocation offline, before any simulator is involved.

Run this first, every time you touch the scene file. It needs no ROS, no
Stonefish and no GPU -- it just reads the .scn and does arithmetic. If the
numbers here are wrong, everything downstream is wrong, and you will spend a
long evening blaming PPO for it.

    python3 -m sauvc_gym.scripts.verify_allocation --scn path/to/my_auv.scn

What "PASS" means here: the declared geometry produces an allocation whose
forward and inverse agree, whose axes are decoupled, and whose authority is
plausible. It does **not** mean the signs match your actual simulator -- only
``identify_allocation.py``, which pulses real thrusters, can tell you that.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from sauvc_gym.allocation import DOF_NAMES, ThrustAllocator
from sauvc_gym.scn_parse import parse_scenario


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scn", required=True, help="path to the VEHICLE .scn file")
    p.add_argument("--robot", default=None, help="robot name, if the file has several")
    p.add_argument("--dofs", nargs="+", default=["surge", "sway", "heave", "yaw"])
    args = p.parse_args(argv)

    np.set_printoptions(precision=3, suppress=True, linewidth=140)

    spec = parse_scenario(args.scn, robot_name=args.robot)
    alloc = ThrustAllocator(spec, action_dofs=tuple(args.dofs))

    print(alloc.describe())
    print("\nallocation matrix B (rows Fx Fy Fz Mx My Mz):")
    print(alloc.B)

    failures: list[str] = []

    # 1. Round trip. allocate() then the forward model must agree, or the
    #    quadratic setpoint inverse is wrong somewhere.
    print("\n--- round trip -------------------------------------------------")
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(200):
        a = rng.uniform(-1, 1, size=len(args.dofs))
        r = alloc.allocate(a)
        err = float(np.max(np.abs(alloc.setpoints_to_wrench(r.setpoints)
                                  - r.wrench_delivered)))
        worst = max(worst, err)
    print(f"max |forward(allocate(a)) - delivered|  = {worst:.2e} N/Nm")
    if worst > 1e-6:
        failures.append("round trip does not close -- check setpoint_from_thrust")

    # 2. Decoupling. A pure command on one axis must not leak into another.
    print("\n--- axis decoupling (unit command on each DOF) ------------------")
    for k, dof in enumerate(args.dofs):
        a = np.zeros(len(args.dofs))
        a[k] = 1.0
        r = alloc.allocate(a)
        w = r.wrench_delivered
        leak = {
            DOF_NAMES[j]: w[j]
            for j in range(6)
            if DOF_NAMES[j] != dof and abs(w[j]) > 1e-3
        }
        status = "clean" if not leak else f"LEAKS {leak}"
        print(f"  {dof:<6} -> {np.round(w, 2)}  {status}")
        if leak:
            failures.append(f"{dof} leaks into {list(leak)}")

    # 3. Saturation must scale, not clip. Ask for more than the vehicle has and
    #    confirm the direction survives.
    print("\n--- saturation preserves direction ------------------------------")
    a = np.ones(len(args.dofs))
    r = alloc.allocate(a)
    req, got = r.wrench_requested, r.wrench_delivered
    idx = [j for j in range(6) if abs(req[j]) > 1e-6]
    if idx:
        ratios = np.array([got[j] / req[j] for j in idx])
        spread = float(ratios.max() - ratios.min())
        print(f"  all-axes-max: scale={r.saturation:.3f} "
              f"per-axis ratios={np.round(ratios, 4)}")
        # Grouped allocation scales the two groups independently, so ratios may
        # form two clusters; within a group they must be identical.
        if spread > 1e-6 and len(set(np.round(ratios, 6))) > 2:
            failures.append("saturation distorts wrench direction")

    # 4. Setpoints must be legal.
    if np.any(np.abs(r.setpoints) > 1.0 + 1e-9):
        failures.append(f"setpoints out of [-1,1]: {r.setpoints}")

    # 5. Sanity on the physics, using the numbers you validated empirically.
    print("\n--- plausibility ------------------------------------------------")
    for t in spec.thrusters:
        if not (20.0 <= t.max_thrust <= 120.0):
            print(f"  ! {t.name}: max thrust {t.max_thrust:.1f} N is outside the "
                  f"20-120 N band expected of a T200-class thruster. "
                  f"Kt={t.thrust_coeff}, w_max={t.max_omega}")
            failures.append(f"{t.name} implausible max thrust")
    print(f"  total vertical authority {alloc.tau_max[2]:.1f} N")
    print(f"  total surge authority    {alloc.tau_max[0]:.1f} N")
    print(f"  total yaw authority      {alloc.tau_max[5]:.1f} Nm")

    print("\n" + "=" * 64)
    if failures:
        print("FAIL")
        for f in failures:
            print(f"  - {f}")
        print("\nDo not train against this. Fix the scene file or the parser first.")
        return 1
    print("PASS -- geometry is self-consistent.")
    print("Next: identify_allocation.py, to confirm the SIGNS against the "
          "running simulator. This script cannot check those.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
