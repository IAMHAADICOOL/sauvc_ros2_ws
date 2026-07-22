#!/usr/bin/env python3
"""csv_to_tum.py — convert odometry.csv / odometry_anchored.csv into TUM trajectory format.

TUM format (one pose per line, whitespace-separated):
    timestamp tx ty tz qx qy qz qw

This is the format `evo`, the TUM RGB-D benchmark tools, and align_and_evaluate.py (this
package) all expect. Most VIO/SLAM pipelines can export it directly (ORB-SLAM3,
VINS-Fusion, OpenVINS, ...); this script exists for the one side of the comparison this
package itself produces — the recorded ground truth.

Usage:
    python3 csv_to_tum.py odometry.csv gt.tum
    python3 csv_to_tum.py odometry_anchored.csv gt_anchored.tum
    python3 csv_to_tum.py my_vio_output.csv est.tum --t-col time --x-col px ...

Column names default to this package's own CSV layout (t, x, y, z, qx, qy, qz, qw) but
are all overridable, so this also works on a differently-named CSV from elsewhere.
"""

import argparse
import csv


def convert(in_path, out_path, cols):
    rows = []
    with open(in_path, newline='') as f:
        reader = csv.DictReader(f)
        missing = [c for c in cols.values() if c not in (reader.fieldnames or [])]
        if missing:
            raise SystemExit(
                f"column(s) {missing} not found in {in_path}; found {reader.fieldnames}. "
                "Override with --t-col/--x-col/... to match your file.")
        for r in reader:
            rows.append((
                float(r[cols['t']]), float(r[cols['x']]), float(r[cols['y']]),
                float(r[cols['z']]), float(r[cols['qx']]), float(r[cols['qy']]),
                float(r[cols['qz']]), float(r[cols['qw']])))

    # Defensive: TUM tools (and align_and_evaluate.py) assume ascending time.
    rows.sort(key=lambda row: row[0])

    with open(out_path, 'w') as f:
        f.write(f"# {in_path} -> TUM (timestamp tx ty tz qx qy qz qw)\n")
        for t, x, y, z, qx, qy, qz, qw in rows:
            f.write(f"{t:.9f} {x:.9f} {y:.9f} {z:.9f} "
                    f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n")

    print(f"wrote {len(rows)} poses -> {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input_csv')
    ap.add_argument('output_tum')
    ap.add_argument('--t-col', default='t')
    ap.add_argument('--x-col', default='x')
    ap.add_argument('--y-col', default='y')
    ap.add_argument('--z-col', default='z')
    ap.add_argument('--qx-col', default='qx')
    ap.add_argument('--qy-col', default='qy')
    ap.add_argument('--qz-col', default='qz')
    ap.add_argument('--qw-col', default='qw')
    args = ap.parse_args()

    cols = {'t': args.t_col, 'x': args.x_col, 'y': args.y_col, 'z': args.z_col,
           'qx': args.qx_col, 'qy': args.qy_col, 'qz': args.qz_col, 'qw': args.qw_col}
    convert(args.input_csv, args.output_tum, cols)


if __name__ == '__main__':
    main()
