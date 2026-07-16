#!/usr/bin/env python3
"""
Converts the raw vehicle CAD OBJ (centimeters, Y-up) into Stonefish-ready meshes:
  my_auv_vis.obj - visual mesh, per-part decimated, hardware culled (~190k faces)
  my_auv_phy.obj - convex-hull physical mesh (collision + hydrodynamics proxy)

WHY PER-PART: global quadric decimation across a multi-body CAD export collapses
edges BETWEEN disconnected parts and shreds the model into shards. This script
splits the mesh into connected components, drops tiny hardware (screws/nuts,
bbox diagonal < 1.5 cm), decimates each remaining part individually, and verifies
each result stays inside its part's bounding box (falling back to the undecimated
part if not).

CAD -> NED: x_ned = x_cad, y_ned = z_cad, z_ned = -y_cad, then cm -> m.
Origin: geometric center of the FULL CAD bounding box (so actuator/sensor
coordinates in my_auv.scn remain valid regardless of culling).

Usage: python3 convert_vehicle_mesh.py <input.obj> <output_dir>
Requires: trimesh, scipy, fast-simplification (pip install)
"""
import sys, time
import numpy as np
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
import fast_simplification as fs

MIN_PART_DIAG_CM = 1.5      # drop parts smaller than this (screws, nuts, washers)
DECIMATE_ABOVE_FACES = 600  # only decimate parts bigger than this
TARGET_REDUCTION = 0.85     # remove ~85% of faces per decimated part


def main(src, outdir):
    t0 = time.time()
    m = trimesh.load(src, force='mesh', process=False)
    m.merge_vertices()
    V, F = np.asarray(m.vertices, dtype=np.float64), np.asarray(m.faces)
    print(f'merged: {len(V)} v, {len(F)} f ({time.time()-t0:.0f}s)')

    e = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(V), len(V)))
    ncomp, vlabel = connected_components(g, directed=False)
    flabel = vlabel[F[:, 0]]
    print(f'components: {ncomp}')

    out_v, out_f = [], []
    kept = dropped = fallback = 0
    counts = np.bincount(flabel, minlength=ncomp)
    for lbl in np.argsort(counts)[::-1]:
        if counts[lbl] == 0:
            continue
        cf = F[flabel == lbl]
        vid = np.unique(cf)
        lut = np.full(len(V), -1, dtype=np.int64)
        lut[vid] = np.arange(len(vid))
        cv, cf = V[vid], lut[cf]
        diag = np.linalg.norm(cv.max(0) - cv.min(0))
        if diag < MIN_PART_DIAG_CM:
            dropped += 1
            continue
        if len(cf) > DECIMATE_ABOVE_FACES:
            try:
                dv, df = fs.simplify(cv, cf, target_reduction=TARGET_REDUCTION)
                ok = (len(df) > 20
                      and np.all(dv.min(0) >= cv.min(0) - 0.05 * diag)
                      and np.all(dv.max(0) <= cv.max(0) + 0.05 * diag))
            except Exception:
                ok = False
            if not ok:
                fallback += 1
                dv, df = cv, cf
        else:
            dv, df = cv, cf
        base = sum(len(v) for v in out_v)
        out_v.append(dv)
        out_f.append(df + base)
        kept += 1

    V2, F2 = np.vstack(out_v), np.vstack(out_f)
    print(f'kept {kept} parts, dropped {dropped} tiny, fallback {fallback}; faces -> {len(F2)}')

    center = (m.bounds[0] + m.bounds[1]) / 2.0
    def to_ned(verts):
        v = verts - center
        return np.column_stack([v[:, 0], v[:, 2], -v[:, 1]]) * 0.01

    # Double-sided visual mesh: CAD shells have inconsistent normals, so single-sided
    # rendering shows see-through patches (backface culling). Duplicating the geometry
    # with reversed winding makes every surface visible from both sides.
    ned = to_ned(V2)
    ned_ds = np.vstack([ned, ned])
    F_ds = np.vstack([F2, F2[:, ::-1] + len(ned)])
    vis = trimesh.Trimesh(vertices=ned_ds, faces=F_ds, process=False)
    vis.export(outdir + '/my_auv_vis.obj', include_normals=True)
    print(f'visual: {len(vis.faces)} faces -> {outdir}/my_auv_vis.obj')

    full = trimesh.Trimesh(vertices=to_ned(V), faces=F, process=False)
    hull = full.convex_hull
    hull.fix_normals()
    hull.export(outdir + '/my_auv_phy.obj', include_normals=True)
    print(f'physical: {len(hull.faces)} faces, volume {hull.volume:.4f} m^3 -> {outdir}/my_auv_phy.obj')
    print('NOTE: hull volume overestimates true displacement; mass/buoyancy come from the')
    print('internal Float/Ballast volumes in my_auv.scn - tune those to your real vehicle.')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
