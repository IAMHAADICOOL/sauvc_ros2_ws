import sys, numpy as np
sys.path.insert(0, '.')
# stub sauvc_sim_bridge.frames so eval_common imports offline
import types
m = types.ModuleType('sauvc_sim_bridge'); mf = types.ModuleType('sauvc_sim_bridge.frames')
mf.ned_to_enu_vec = lambda v: np.array([v[1], v[0], -v[2]])
mf.frd_to_flu_vec = lambda v: np.array([v[0], -v[1], -v[2]])
mf.ned_frd_quat_to_enu_flu = lambda q: q
sys.modules['sauvc_sim_bridge'] = m; sys.modules['sauvc_sim_bridge.frames'] = mf

from sauvc_flow_eval.eval_common import (
    enu_to_ned_vec, ned_to_enu_world, depth_to_world_z,
    gt_world_to_compare, to_compare_frame_world, PositionIntegrator)

ok = True
def check(n, c):
    global ok; ok &= bool(c); print(f"  [{'PASS' if c else 'FAIL'}] {n}")

print("Frame conversions are involutions & magnitude-preserving")
v = np.array([3.0, -7.0, 11.0])
check("enu<->ned self-inverse", np.allclose(enu_to_ned_vec(enu_to_ned_vec(v)), v))
check("magnitude preserved", np.isclose(np.linalg.norm(enu_to_ned_vec(v)), np.linalg.norm(v)))

print("\nDepth sign per frame")
check("NED depth 1.2 -> z=+1.2 (down positive)", depth_to_world_z(1.2, 'ned') == 1.2)
check("ENU depth 1.2 -> z=-1.2 (up positive)", depth_to_world_z(1.2, 'enu') == -1.2)

print("\nGround truth passthrough in NED, converted in ENU")
gt_ned = np.array([2.0, 5.0, 0.3])
check("ned: gt untouched", np.allclose(gt_world_to_compare(gt_ned, 'ned'), gt_ned))
check("enu: gt N->E swaps x/y, flips z", np.allclose(gt_world_to_compare(gt_ned, 'enu'), [5.0, 2.0, -0.3]))

print("\nEstimate (ENU) -> compare frame")
est_enu = np.array([1.0, 2.0, -0.5])
check("enu: estimate untouched", np.allclose(to_compare_frame_world(est_enu, 'enu'), est_enu))
check("ned: estimate E->N swaps x/y flips z", np.allclose(to_compare_frame_world(est_enu, 'ned'), [2.0, 1.0, 0.5]))

print("\nPositionIntegrator: straight-line dead reckoning")
pi = PositionIntegrator()
# move at 1 m/s forward (body x), yaw=0, for 10 steps of 0.1s -> ~1.0 m in world x
t = 0.0
for _ in range(11):
    x, y = pi.update(1.0, 0.0, 0.0, t); t += 0.1
check(f"~1.0 m after 1 s at 1 m/s (x={x:.3f})", abs(x - 1.0) < 0.05 and abs(y) < 1e-6)

pi.reset()
# yaw=90deg, body-forward should map to world +y
t = 0.0
for _ in range(11):
    x, y = pi.update(1.0, 0.0, np.pi/2, t); t += 0.1
check(f"yaw=90: body-forward -> world +y (y={y:.3f}, x={x:.3f})", abs(y - 1.0) < 0.05 and abs(x) < 1e-6)

pi.reset()
check("absurd dt ignored", pi.update(1,0,0,0.0) == (0.0,0.0) and pi.update(1,0,0,5.0)[0] == 0.0)

print("\n" + ("ALL EVAL_COMMON TESTS PASSED" if ok else "FAILED"))
sys.exit(0 if ok else 1)
