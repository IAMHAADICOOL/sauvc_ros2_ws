#!/usr/bin/env python3
"""
offline_flow_test.py — Run FlowVelocityEstimator on a recorded video, no ROS required.

Usage:
  python3 offline_flow_test.py video.mp4 --fx 700 --fy 700 --altitude 1.0
  python3 offline_flow_test.py video.mp4 --fx 700 --fy 700 --altitude 1.0 \
          --gyro gyro.csv          # csv: t_sec, wx, wy, wz  (body rad/s)

Outputs:
  - prints per-second velocity stats and total integrated distance
  - saves trajectory.png (top-down integrated path) and velocity.png

This is Phase 3 of the README. Do the 5 m straight-push test with this script:
integrated distance vs tape measure = your scale error.
"""

import argparse
import csv
import math
import sys

import numpy as np
import cv2

import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "sauvc_localization"))
from flow_core import FlowVelocityEstimator


def load_gyro(path):
    ts, w = [], []
    with open(path) as f:
        for row in csv.reader(f):
            if not row or row[0].startswith('#'):
                continue
            vals = [float(x) for x in row[:4]]
            ts.append(vals[0])
            w.append(vals[1:4])
    return np.array(ts), np.array(w)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('video')
    ap.add_argument('--fx', type=float, required=True)
    ap.add_argument('--fy', type=float, required=True)
    ap.add_argument('--cx', type=float, default=None)
    ap.add_argument('--cy', type=float, default=None)
    ap.add_argument('--altitude', type=float, default=1.0,
                    help='constant altitude above floor in meters')
    ap.add_argument('--gyro', default=None, help='csv: t, wx, wy, wz (body frame)')
    ap.add_argument('--yaw-rate', action='store_true',
                    help='integrate gyro wz for heading while plotting the path')
    ap.add_argument('--show', action='store_true', help='display tracking overlay live')
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f'cannot open {args.video}')
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cx = args.cx if args.cx is not None else W / 2
    cy = args.cy if args.cy is not None else H / 2
    print(f'{W}x{H} @ {fps:.1f} fps, fx={args.fx} fy={args.fy}')

    est = FlowVelocityEstimator(args.fx, args.fy, cx, cy)

    gyro_t = gyro_w = None
    if args.gyro:
        gyro_t, gyro_w = load_gyro(args.gyro)

    dt = 1.0 / fps
    t = 0.0
    x = y = yaw = 0.0
    xs, ys, ts, vxs, vys, nfeat = [0.0], [0.0], [0.0], [], [], []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t += dt
        wx = wy = wz = 0.0
        if gyro_t is not None:
            i = np.searchsorted(gyro_t, t)
            i = min(max(i, 0), len(gyro_t) - 1)
            wx, wy, wz = gyro_w[i]
        # body gyro -> camera frame (same default convention as the ROS node)
        out = est.process(frame, dt, (-wy, -wx), args.altitude)
        if out is None:
            continue
        vx, vy = out['vx'], out['vy']
        if args.yaw_rate:
            yaw += wz * dt
        # integrate in a world frame rotated by yaw (yaw=0 if no gyro given)
        x += (vx * math.cos(yaw) - vy * math.sin(yaw)) * dt
        y += (vx * math.sin(yaw) + vy * math.cos(yaw)) * dt
        xs.append(x); ys.append(y); ts.append(t)
        vxs.append(vx); vys.append(vy); nfeat.append(out['n_inliers'])

        if args.show:
            cv2.putText(frame, f'v=({vx:+.2f},{vy:+.2f}) m/s  n={out["n_inliers"]}',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('flow', frame)
            if cv2.waitKey(1) == 27:
                break

    dist = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
    net = math.hypot(xs[-1], ys[-1])
    print(f'\nframes with estimate: {len(vxs)}')
    if vxs:
        print(f'mean |v| = {np.mean(np.hypot(vxs, vys)):.3f} m/s   '
              f'stationary-noise check: std vx={np.std(vxs):.3f}, vy={np.std(vys):.3f} m/s')
        print(f'median inlier features = {np.median(nfeat):.0f}')
    print(f'path length = {dist:.2f} m,  net displacement = {net:.2f} m')
    print('=> for the 5 m push test compare "net displacement" to your tape measure')

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(xs, ys, '-'); ax.plot(0, 0, 'go'); ax.plot(xs[-1], ys[-1], 'rx')
        ax.set_xlabel('x fwd [m]'); ax.set_ylabel('y left [m]')
        ax.set_aspect('equal'); ax.grid(True)
        ax.set_title(f'integrated path  ({dist:.1f} m travelled)')
        fig.savefig('trajectory.png', dpi=120)
        fig2, ax2 = plt.subplots(figsize=(8, 3))
        ax2.plot(ts[1:], vxs, label='vx'); ax2.plot(ts[1:], vys, label='vy')
        ax2.set_xlabel('t [s]'); ax2.set_ylabel('m/s'); ax2.legend(); ax2.grid(True)
        fig2.savefig('velocity.png', dpi=120)
        print('wrote trajectory.png and velocity.png')
    except ImportError:
        print('matplotlib not installed — skipping plots')


if __name__ == '__main__':
    main()
