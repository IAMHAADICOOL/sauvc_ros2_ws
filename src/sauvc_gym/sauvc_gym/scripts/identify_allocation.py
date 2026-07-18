#!/usr/bin/env python3
"""
Measure the allocation matrix from the running simulator, one thruster at a time.

Why this exists
---------------
The last time this project trusted a thrust model on paper, the coefficient was
wrong by nearly three orders of magnitude, and the time before that a
left-handed propeller sign turned "swim forward" into "topple over". Both were
found by running an experiment, not by reading a document. The same applies to
the allocation: ``scn_parse`` reads the geometry out of the XML and applies a
*documented* rule for how ``right`` and ``inverted_setpoint`` interact, and that
rule is an assumption until measured.

So: pulse each thruster alone, watch what the vehicle actually does, and fit the
column. Then compare against what the scene file claims. Any column that
disagrees in sign is a bug you would otherwise have discovered as "PPO won't
converge".

Method
------
For thruster i, command setpoint +u for a short pulse from rest and integrate
the resulting body-frame linear and angular acceleration. With mass and inertia
unknown, absolute Newtons are not recoverable from kinematics alone -- but the
*direction* and *relative magnitude* of each column are, and those are what
carry the sign errors. So we report each measured column normalised, and check
its correlation against the declared column.

Usage
-----
Start the sim, put the vehicle somewhere with room around it, then::

    python3 -m sauvc_gym.scripts.identify_allocation --scn my_auv.scn -o measured.yaml

Run this in open water, away from walls and the floor: a thruster pulse that
pushes the hull into the bottom measures the floor, not the thruster.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from sauvc_gym.allocation import ThrustAllocator
from sauvc_gym.ros_link import RosLink
from sauvc_gym.scn_parse import parse_scenario


def settle(link: RosLink, seconds: float = 3.0, tol: float = 0.02) -> bool:
    """Zero thrust and wait for the vehicle to stop moving."""
    link.stop()
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        s = link.get_state()
        if (np.linalg.norm(s.lin_vel_body) < tol
                and np.linalg.norm(s.ang_vel_body) < tol):
            return True
        time.sleep(0.05)
    return False


def pulse_thruster(link: RosLink, index: int, n: int, level: float,
                   duration: float) -> np.ndarray:
    """Fire one thruster from rest; return measured (dv, dw) in the body frame.

    The pulse is short and the level modest on purpose. Long pulses let
    hydrodynamic drag dominate and the vehicle start rotating, which couples the
    axes and ruins the measurement. We want the acceleration at t=0, where drag
    is still zero because the velocity is.
    """
    settle(link)
    s0 = link.get_state()

    u = np.zeros(n)
    u[index] = level
    link.send_setpoints(u)

    t0 = time.monotonic()
    time.sleep(duration)
    s1 = link.get_state()
    dt = time.monotonic() - t0
    link.stop()

    dv = (s1.lin_vel_body - s0.lin_vel_body) / dt
    dw = (s1.ang_vel_body - s0.ang_vel_body) / dt
    return np.concatenate([dv, dw])


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scn", required=True)
    p.add_argument("--robot", default="sauvc_auv")
    p.add_argument("--level", type=float, default=0.4, help="pulse setpoint")
    p.add_argument("--duration", type=float, default=0.6, help="pulse length [s]")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("-o", "--output", default=None, help="write measured B to YAML")
    args = p.parse_args(argv)

    np.set_printoptions(precision=3, suppress=True, linewidth=140)

    spec = parse_scenario(args.scn, robot_name=args.robot)
    alloc = ThrustAllocator(spec)
    link = RosLink(robot_name=args.robot, setpoint_topic=spec.setpoint_topic,
                   n_thrusters=spec.n_thrusters)

    if not link.wait_for_data(timeout=30.0):
        print(f"No odometry on {link.odom_topic}. Is the simulator running?",
              file=sys.stderr)
        return 1

    s = link.get_state()
    print(f"vehicle at ({s.position[0]:+.2f}, {s.position[1]:+.2f}, "
          f"{s.position[2]:+.2f})")
    if s.depth < 0.4:
        print("! shallow -- pulses may breach the surface and corrupt the fit")

    measured = np.zeros((6, spec.n_thrusters))
    print(f"\npulsing {spec.n_thrusters} thrusters, {args.repeats} reps each "
          f"at u={args.level} for {args.duration}s\n")

    for i, t in enumerate(spec.thrusters):
        reps = [pulse_thruster(link, i, spec.n_thrusters, args.level, args.duration)
                for _ in range(args.repeats)]
        col = np.mean(reps, axis=0)
        spread = np.std(reps, axis=0)
        measured[:, i] = col
        noisy = " NOISY" if np.max(spread) > 0.3 * (np.max(np.abs(col)) + 1e-9) else ""
        print(f"  {t.name:<6} accel={np.round(col, 3)}{noisy}")

    print("\n--- measured vs declared ---------------------------------------")
    print("Each column normalised; comparing DIRECTION, not magnitude.\n")

    bad: list[str] = []
    for i, t in enumerate(spec.thrusters):
        m = measured[:, i]
        d = alloc.B[:, i]
        nm, nd = np.linalg.norm(m), np.linalg.norm(d)
        if nm < 1e-6:
            print(f"  {t.name:<6} NO RESPONSE -- thruster not moving the vehicle")
            bad.append(f"{t.name}: no response")
            continue
        # Compare only force direction; the moment rows involve the inertia
        # tensor, which kinematics alone cannot invert.
        cos = float(np.dot(m[:3] / nm, d[:3] / nd)) if nd > 1e-9 else 0.0
        verdict = "ok" if cos > 0.8 else ("SIGN FLIPPED" if cos < -0.5 else "MISMATCH")
        print(f"  {t.name:<6} cos(measured, declared) = {cos:+.3f}  {verdict}")
        if cos <= 0.8:
            bad.append(f"{t.name}: cos={cos:+.3f}")

    link.close()

    if args.output:
        import yaml

        with open(args.output, "w") as fh:
            yaml.safe_dump(
                {
                    "robot": args.robot,
                    "thrusters": spec.names,
                    "measured_accel_columns": measured.tolist(),
                    "declared_B": alloc.B.tolist(),
                    "note": "measured columns are accelerations, not wrench; "
                            "use for sign checking only",
                },
                fh,
            )
        print(f"\nwrote {args.output}")

    print("\n" + "=" * 64)
    if bad:
        print("FAIL -- the simulator does not agree with the scene file:")
        for b in bad:
            print(f"  - {b}")
        print("\nA flipped sign here is the 'vehicle doesn't move then topples' "
              "bug. Fix inverted_setpoint / right in the .scn before training.")
        return 1
    print("PASS -- every thruster pushes the way the scene file says it does.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
