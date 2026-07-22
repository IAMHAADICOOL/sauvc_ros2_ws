#!/usr/bin/env python3
"""align_and_evaluate.py — align an estimated trajectory to ground truth and score it.

Two TUM trajectories in, one alignment + ATE (Absolute Trajectory Error) report out.
Pure numpy, no ROS — same spirit as this project's own EKF/GTSAM estimators (see
sauvc_flow_eval/estimators/): a transparent, dependency-light core you can read
top-to-bottom, rather than a black box.

WHY A LEAST-SQUARES ALIGNMENT OVER THE WHOLE TRAJECTORY, NOT JUST THE FIRST POSE
-----------------------------------------------------------------------------------
A general VIO/SLAM pipeline defines its world frame from its first camera pose, which is
rotated arbitrarily relative to ground truth's NED/ENU world (IMU-aided pipelines align
gravity but not heading; pure monocular aligns nothing). Monocular pipelines also have no
absolute scale. Subtracting just the first pose (what odometry_anchored.csv / this
project's own flow_eval_node._anchor do, correctly, for THEIR AHRS-referenced estimators)
fixes none of that, and is fragile besides — a single noisy first pose throws off every
later comparison. The standard fix (Umeyama, 1991; what `evo`, the TUM RGB-D benchmark,
and the KITTI devkit all use under the hood) is a rigid or similarity transform fit by
least squares over every matched pose pair at once:

    dst_i ~= s * R @ src_i + t          for all matched i

    --align       (SE(3), s fixed at 1)   for pipelines with metric scale (stereo,
                                          RGB-D, VIO with IMU/baseline)
    --align-scale (Sim(3), s solved for)  for monocular-only VO/SLAM with unknown scale

USAGE
    python3 align_and_evaluate.py gt.tum est.tum --align
    python3 align_and_evaluate.py gt.tum est.tum --align-scale --plot out.png
"""

import argparse

import numpy as np


def load_tum(path):
    """Return (t[N], pos[N,3], quat_xyzw[N,4]), sorted by timestamp."""
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = [float(v) for v in line.split()]
            if len(parts) != 8:
                raise ValueError(f'{path}: expected 8 columns (t tx ty tz qx qy qz qw), '
                                 f'got {len(parts)}: {line!r}')
            rows.append(parts)
    rows.sort(key=lambda r: r[0])
    arr = np.array(rows, dtype=float)
    return arr[:, 0], arr[:, 1:4], arr[:, 4:8]


def associate(gt_t, est_t, max_diff=0.02):
    """Greedy nearest-neighbour 1:1 timestamp matching. Returns list of (i_gt, i_est).

    Good enough for scoring a handful of ~10 trajectories; for rigorous benchmarking
    (bag-scale datasets, publication numbers) prefer `evo`'s association, which handles
    edge cases (duplicate timestamps, large clock offsets) more carefully. This is
    intentionally simple and readable instead.
    """
    used = np.zeros(len(gt_t), dtype=bool)
    pairs = []
    for j, te in enumerate(est_t):
        idx = int(np.searchsorted(gt_t, te))
        candidates = [k for k in (idx - 1, idx) if 0 <= k < len(gt_t) and not used[k]]
        if not candidates:
            continue
        best = min(candidates, key=lambda k: abs(gt_t[k] - te))
        if abs(gt_t[best] - te) <= max_diff:
            pairs.append((best, j))
            used[best] = True
    return pairs


def umeyama_alignment(src, dst, with_scale=False):
    """Umeyama (1991): find R, t, s minimizing sum ||dst_i - (s*R@src_i + t)||^2.

    src, dst: (N,3) arrays of CORRESPONDING points (already associated/ordered).
    Returns (R (3,3), t (3,), s (float)). s is always 1.0 if with_scale=False.
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    src_c, dst_c = src - mu_src, dst - mu_dst

    sigma2 = (src_c ** 2).sum() / n
    cov = (dst_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0.0:
        S[2, 2] = -1.0

    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / sigma2) if with_scale else 1.0
    t = mu_dst - s * (R @ mu_src)
    return R, t, s


def apply_transform(pts, R, t, s):
    return (s * (R @ pts.T)).T + t


def ate(aligned_src, dst):
    errors = np.linalg.norm(aligned_src - dst, axis=1)
    return {
        'rmse': float(np.sqrt(np.mean(errors ** 2))),
        'mean': float(np.mean(errors)),
        'median': float(np.median(errors)),
        'std': float(np.std(errors)),
        'min': float(np.min(errors)),
        'max': float(np.max(errors)),
        'n': int(len(errors)),
    }, errors


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('gt_tum', help='ground truth trajectory, TUM format')
    ap.add_argument('est_tum', help='estimated trajectory (your VIO/SLAM output), TUM format')
    ap.add_argument('--max-diff', type=float, default=0.02,
                    help='max timestamp gap for association [s] (default 0.02)')
    align_group = ap.add_mutually_exclusive_group()
    align_group.add_argument('--align', action='store_true',
                             help='SE(3) alignment (rotation + translation, scale=1) — '
                                  'use for stereo/RGB-D/VIO-with-IMU (metric scale)')
    align_group.add_argument('--align-scale', action='store_true',
                             help='Sim(3) alignment (rotation + translation + scale) — '
                                  'use for MONOCULAR-only VO/SLAM (unknown scale)')
    ap.add_argument('--plot', default=None, help='save a top-down XY comparison PNG here')
    args = ap.parse_args()

    gt_t, gt_pos, _ = load_tum(args.gt_tum)
    est_t, est_pos, _ = load_tum(args.est_tum)
    print(f'ground truth: {len(gt_t)} poses, span {gt_t[-1] - gt_t[0]:.1f}s')
    print(f'estimate:     {len(est_t)} poses, span {est_t[-1] - est_t[0]:.1f}s')

    pairs = associate(gt_t, est_t, max_diff=args.max_diff)
    if len(pairs) < 3:
        raise SystemExit(
            f'only {len(pairs)} matched poses within {args.max_diff}s — cannot align. '
            'Check that both files share a time base (see the sync notes in meta.yaml) '
            'and try a larger --max-diff.')
    print(f'matched {len(pairs)} pose pairs (max_diff={args.max_diff}s)')

    gi = np.array([p[0] for p in pairs])
    ei = np.array([p[1] for p in pairs])
    dst = gt_pos[gi]     # ground truth points
    src = est_pos[ei]    # estimate points, to be aligned onto dst

    with_scale = bool(args.align_scale)
    if not args.align and not args.align_scale:
        print('no --align/--align-scale given — defaulting to SE(3) (scale=1). '
             'Pass --align-scale if your pipeline is monocular-only.')
    R, t, s = umeyama_alignment(src, dst, with_scale=with_scale)
    aligned = apply_transform(src, R, t, s)

    metrics, errors = ate(aligned, dst)
    print(f"\nalignment: {'Sim(3), scale=' + f'{s:.4f}' if with_scale else 'SE(3), scale=1.0'}")
    print('ATE (m):')
    for k in ('rmse', 'mean', 'median', 'std', 'min', 'max'):
        print(f'  {k:>7s}: {metrics[k]:.4f}')
    print(f'  matched: {metrics["n"]}')

    if args.plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except Exception as e:
            print(f'--plot requested but matplotlib is unavailable ({e}); skipping plot.')
        else:
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.plot(gt_pos[:, 0], gt_pos[:, 1], label='ground truth', linewidth=2)
            ax.plot(aligned[:, 0], aligned[:, 1], label='estimate (aligned)',
                    linewidth=1.5, linestyle='--')
            ax.set_xlabel('x [m]'); ax.set_ylabel('y [m]')
            ax.set_title(f'ATE RMSE = {metrics["rmse"]:.3f} m  (n={metrics["n"]})')
            ax.legend(); ax.axis('equal'); ax.grid(True, alpha=0.3)
            fig.savefig(args.plot, dpi=150, bbox_inches='tight')
            print(f'saved plot -> {args.plot}')


if __name__ == '__main__':
    main()
