#!/usr/bin/env python3
"""estimate_covariance.py — turn a column of logged stationary samples into the
variance number you paste into a covariance field or node parameter.

Usage:
  # 1. Log real data while the sensor sits still (or in a fixed motion for the yaw-drift
  #    case). Easiest capture, no custom code needed:
  ros2 topic echo --csv /imu/data/angular_velocity > gyro.csv
  # (columns: x, y, z — stop it after ~60s with Ctrl-C)

  # 2. Run this on a column:
  python3 estimate_covariance.py gyro.csv --col 2 --label "gyro wz"

Also handles the "yaw drift" case, where what you actually have is a DRIFT RATE
(deg/min) rather than a stationary noise sample, via --drift-rate-deg-per-min and
--window-s (how long between corrections the filter will realistically run without a
lane-line/landmark update) — converts drift into an equivalent variance so the number
means "how much yaw uncertainty accumulates in one filter cycle without correction."
"""
import argparse
import csv
import math
import sys
import numpy as np


def variance_from_samples(path, col):
    vals = []
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith('#'):
                continue
            try:
                vals.append(float(row[col]))
            except (ValueError, IndexError):
                continue
    if len(vals) < 10:
        sys.exit(f'only {len(vals)} usable rows — log more data (aim for 500+ samples)')
    arr = np.array(vals)
    return arr, float(np.var(arr)), float(np.mean(arr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv_file', nargs='?')
    ap.add_argument('--col', type=int, default=0, help='0-indexed column to use')
    ap.add_argument('--label', default='signal')
    ap.add_argument('--drift-rate-deg-per-min', type=float, default=None,
                    help='use instead of a csv for yaw: converts a measured drift '
                         'rate into an equivalent variance per correction window')
    ap.add_argument('--window-s', type=float, default=5.0,
                    help='how often you expect a real correction (e.g. lane-line hit '
                         'rate) — the drift accumulated in this window becomes the sigma')
    args = ap.parse_args()

    if args.drift_rate_deg_per_min is not None:
        drift_rad = math.radians(args.drift_rate_deg_per_min) / 60.0   # rad/s
        sigma = drift_rad * args.window_s
        var = sigma ** 2
        print(f'{args.label}: drift {args.drift_rate_deg_per_min} deg/min over a '
              f'{args.window_s}s window -> sigma={math.degrees(sigma):.3f} deg '
              f'({sigma:.5f} rad)  ->  variance = {var:.7f} rad^2')
        return

    if not args.csv_file:
        sys.exit('need either a csv_file or --drift-rate-deg-per-min')

    arr, var, mean = variance_from_samples(args.csv_file, args.col)
    std = math.sqrt(var)
    print(f'{args.label}: n={len(arr)}  mean={mean:.5f}  std={std:.5f}  variance={var:.7f}')
    print(f'  -> paste variance = {var:.6g} into the covariance field for {args.label}')
    if abs(mean) > 3 * std:
        print(f'  NOTE: mean is >3-sigma from 0 — check for a bias/offset that needs '
              f'zeroing before this number is meaningful (e.g. gyro bias, sensor tare)')


if __name__ == '__main__':
    main()
