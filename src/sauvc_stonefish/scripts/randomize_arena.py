#!/usr/bin/env python3
"""
Randomize the SAUVC finals arena, matching the official top-view zones:

  - ORANGE FLARE: anywhere in the band 4-8 m from the starting wall
                  -> x in [-8.5, -4.5], y in [-7.0, 7.0]
  - COLORED FLARES (R/Y/B + golf balls): anywhere in the band between the
                  orange zone and the gate line -> x in [-4.4, 3.9], y in [-7.0, 7.0],
                  with >= 1.5 m separation between flares and from the orange flare
  - GATE: anywhere along the gate line x = 4.4 -> center y in [-6.0, 6.0]

Heights are recomputed from the sloped floor: d(x) = 1.6 - 0.032*|x|.
The pinger drum / drums / starting zone are fixed by the rulebook and untouched.

SEEDING: --seed N gives a fully reproducible layout (same N = same arena, across
machines). Different N = a new competition draw.

Usage:
  python3 randomize_arena.py --seed 42                       # writes sauvc_finals_seed42.scn next to the input
  python3 randomize_arena.py --seed 42 --in-place            # overwrites sauvc_finals.scn
  python3 randomize_arena.py --seed 7 -i input.scn -o out.scn

Or use the launch wrapper (regenerates into /tmp and starts the sim):
  ros2 launch sauvc_stonefish sauvc_finals_random.launch.py seed:=42
"""
import argparse
import math
import random
import re
import sys


def depth(x):
    return 1.6 - 0.032 * abs(x)


def set_world_transform(text, body_name, x, y, z, tag='static|dynamic|comm'):
    """Replace the world_transform xyz of a named body, keeping rpy."""
    pat = re.compile(
        r'(<(?:' + tag + r') name="' + re.escape(body_name) +
        r'".*?<world_transform xyz=")[^"]*(")', re.S)
    new, n = pat.subn(lambda m: f'{m.group(1)}{x:.3f} {y:.3f} {z:.4f}{m.group(2)}', text)
    if n != 1:
        sys.exit(f'ERROR: expected exactly 1 transform for {body_name}, found {n}')
    return new


def randomize(text, seed):
    rng = random.Random(seed)

    # ---- Orange flare: x in [-8.5,-4.5], y in [-7,7]; tip reaches the surface ----
    ox = rng.uniform(-8.5, -4.5)
    oy = rng.uniform(-7.0, 7.0)
    od = depth(ox)
    # cylinder height is fixed at 1.4 in the file; center z so the base sits on the floor
    text = set_world_transform(text, 'OrangeFlare', ox, oy, od - 0.70)

    # ---- Colored flares: band [-4.4, 3.9] x [-7, 7], min separation 1.5 m ----
    placed = [(ox, oy)]
    flare_pos = {}
    for color in ('Blue', 'Red', 'Yellow'):
        for _ in range(1000):
            fx = rng.uniform(-4.4, 3.9)
            fy = rng.uniform(-7.0, 7.0)
            if all(math.hypot(fx - px, fy - py) >= 1.5 for px, py in placed):
                break
        else:
            sys.exit('ERROR: could not place flares with separation - relax limits')
        placed.append((fx, fy))
        flare_pos[color] = (fx, fy)
        d = depth(fx)
        text = set_world_transform(text, f'Flare{color}', fx, fy, d - 0.0125)
        text = set_world_transform(text, f'GolfBall{color}', fx, fy, d - 0.832)

    # ---- Gate along the line x = 4.4: center y in [-6, 6] ----
    gy = rng.uniform(-6.0, 6.0)
    gd = depth(4.4)
    text = set_world_transform(text, 'GatePostPort', 4.4, gy - 0.75, gd - 0.5 + 0.006)
    text = set_world_transform(text, 'GatePostStbd', 4.4, gy + 0.75, gd - 0.5 + 0.006)
    text = set_world_transform(text, 'GateTopBar', 4.4, gy, gd - 1.0 + 0.006)

    layout = {'seed': seed, 'orange_flare': (round(ox, 2), round(oy, 2)),
              **{f'{c.lower()}_flare': (round(x, 2), round(y, 2))
                 for c, (x, y) in flare_pos.items()},
              'gate_center': (4.4, round(gy, 2))}
    return text, layout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('-i', '--input', default=None,
                    help='input .scn (default: sauvc_finals.scn next to this script)')
    ap.add_argument('-o', '--output', default=None)
    ap.add_argument('--in-place', action='store_true')
    args = ap.parse_args()

    import os
    if args.input is None:
        args.input = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  '..', 'scenarios', 'sauvc_finals.scn')
    text = open(args.input).read()
    text, layout = randomize(text, args.seed)

    if args.in_place:
        out = args.input
    elif args.output:
        out = args.output
    else:
        base, ext = os.path.splitext(args.input)
        out = f'{base}_seed{args.seed}{ext}'
    open(out, 'w').write(text)
    print('layout:', layout)
    print('written:', out)


if __name__ == '__main__':
    main()
