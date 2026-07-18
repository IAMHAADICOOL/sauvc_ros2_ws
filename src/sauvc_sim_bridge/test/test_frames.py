import numpy as np
from sauvc_sim_bridge.frames import *

def rpy_to_quat(r, p, y):
    """ZYX intrinsic rpy -> (x,y,z,w). Reference implementation for tests."""
    cr, sr = np.cos(r/2), np.sin(r/2)
    cp, sp = np.cos(p/2), np.sin(p/2)
    cy, sy = np.cos(y/2), np.sin(y/2)
    return np.array([sr*cp*cy - cr*sp*sy,
                     cr*sp*cy + sr*cp*sy,
                     cr*cp*sy - sr*sp*cy,
                     cr*cp*cy + sr*sp*sy])

def quat_to_rpy(q):
    x, y, z, w = q
    r = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
    p = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
    yw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
    return np.degrees([r, p, yw])

def quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w),   2*(x*z+y*w)],
        [2*(x*y+z*w),   1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w),   2*(y*z+x*w),   1-2*(x*x+y*y)]])

def same_rot(qa, qb, tol=1e-9):
    """Quaternions double-cover: q and -q are the same rotation."""
    return min(np.linalg.norm(qa-qb), np.linalg.norm(qa+qb)) < tol

ok = True
def check(name, cond):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")

print("Fixed rotations are unit quaternions and involutions")
check("Q_NED_ENU unit", abs(np.linalg.norm(Q_NED_ENU)-1) < 1e-12)
check("Q_FRD_FLU unit", abs(np.linalg.norm(Q_FRD_FLU)-1) < 1e-12)
check("Q_NED_ENU self-inverse", same_rot(quat_mul(Q_NED_ENU, Q_NED_ENU), np.array([0,0,0,1.0])))
check("Q_FRD_FLU self-inverse", same_rot(quat_mul(Q_FRD_FLU, Q_FRD_FLU), np.array([0,0,0,1.0])))

print("\nQuaternions match their matrix forms")
check("Q_NED_ENU matrix agrees", np.allclose(quat_to_R(Q_NED_ENU), rot_matrix_ned_enu()))
check("Q_FRD_FLU matrix agrees", np.allclose(quat_to_R(Q_FRD_FLU), rot_matrix_frd_flu()))

print("\nVector maps: known answers")
# North in NED = (1,0,0). In ENU, North = +y = (0,1,0).
check("NED North -> ENU +y", np.allclose(ned_to_enu_vec([1,0,0]), [0,1,0]))
# East in NED = (0,1,0). In ENU, East = +x = (1,0,0).
check("NED East -> ENU +x", np.allclose(ned_to_enu_vec([0,1,0]), [1,0,0]))
# Down in NED = (0,0,1). In ENU that is -z.
check("NED Down -> ENU -z", np.allclose(ned_to_enu_vec([0,0,1]), [0,0,-1]))
check("NED->ENU involution", np.allclose(ned_to_enu_vec(ned_to_enu_vec([3,-7,11])), [3,-7,11]))
check("FRD Right -> FLU -y", np.allclose(frd_to_flu_vec([0,1,0]), [0,-1,0]))
check("FRD Down -> FLU -z", np.allclose(frd_to_flu_vec([0,0,1]), [0,0,-1]))
check("FRD->FLU involution", np.allclose(frd_to_flu_vec(frd_to_flu_vec([3,-7,11])), [3,-7,11]))

print("\nOrientation: identity NED/FRD (nose North, level) -> ENU yaw = +90 deg")
q = ned_frd_quat_to_enu_flu(np.array([0,0,0,1.0]))
rpy = quat_to_rpy(q)
check(f"rpy = {np.round(rpy,6)} -> yaw 90, roll 0, pitch 0",
      np.allclose(rpy, [0,0,90], atol=1e-6))

print("\nOrientation: NED yaw=+90 (nose East) -> ENU yaw = 0 (nose East = +x)")
q = ned_frd_quat_to_enu_flu(rpy_to_quat(0,0,np.pi/2))
check(f"rpy = {np.round(quat_to_rpy(q),6)}", np.allclose(quat_to_rpy(q), [0,0,0], atol=1e-6))

print("\nOrientation: NED roll=+30 (starboard down) -> ENU roll = +30, NOT -30.")
print("  Roll is SIGN-INVARIANT: negating both y and z preserves rotations about x.")
print("  FRD roll+30 = right-side-down; FLU roll+30 = left-side-up = right-side-down.")
print("  Same physical attitude, same sign. Pitch and yaw DO flip; roll does not.")
q = ned_frd_quat_to_enu_flu(rpy_to_quat(np.radians(30),0,0))
rpy = quat_to_rpy(q)
check(f"rpy = {np.round(rpy,6)}", np.allclose(rpy, [30,0,90], atol=1e-6))

print("\nOrientation: NED pitch=+20 (nose up in NED) -> ENU pitch = -20")
q = ned_frd_quat_to_enu_flu(rpy_to_quat(0,np.radians(20),0))
rpy = quat_to_rpy(q)
check(f"rpy = {np.round(rpy,6)}", np.allclose(rpy, [0,-20,90], atol=1e-6))

print("\nConsistency: rotating a body vector then converting == converting then rotating")
# For a random attitude, the world-frame vector must agree via both paths.
rng = np.random.default_rng(7)
for i in range(2000):
    q_ned = quat_normalize(rng.normal(size=4))
    v_frd = rng.normal(size=3)
    # Path A: rotate in NED/FRD, then convert the resulting world vector.
    v_ned = quat_to_R(q_ned) @ v_frd
    a = ned_to_enu_vec(v_ned)
    # Path B: convert body vector and attitude, then rotate in ENU/FLU.
    q_enu = ned_frd_quat_to_enu_flu(q_ned)
    b = quat_to_R(q_enu) @ frd_to_flu_vec(v_frd)
    if not np.allclose(a, b, atol=1e-9):
        check(f"commutes at sample {i}", False)
        break
else:
    check("commutes over 2000 random attitudes/vectors", True)

print("\nCovariance: diagonal is permuted, signs square away")
C = np.diag([0.01, 0.04, 0.09])
check("FRD->FLU leaves diagonal untouched", np.allclose(cov3_frd_to_flu(C), C))
check("NED->ENU swaps xx and yy", np.allclose(np.diag(cov3_ned_to_enu(C)), [0.04, 0.01, 0.09]))
print("     ^ this is the trap named in the docstring: a diagonal covariance")
print("       cannot reveal a sign error, only an axis-assignment error.")

print("\nCovariance: off-diagonal DOES flip sign")
C2 = np.array([[1.0, 0.5, 0.0],[0.5, 2.0, 0.3],[0.0, 0.3, 3.0]])
out = cov3_frd_to_flu(C2)
check("xy term negated by FRD->FLU", np.isclose(out[0,1], -0.5))
check("yz term preserved by FRD->FLU", np.isclose(out[1,2], 0.3))
check("covariance stays symmetric", np.allclose(out, out.T))
check("covariance stays PSD", np.all(np.linalg.eigvalsh(out) > 0))

print("\nFlat-9 (ROS row-major) round-trips shape")
flat = np.diag([0.01,0.04,0.09]).reshape(9)
r = cov3_ned_to_enu(flat)
check("flat in -> flat out", r.shape == (9,))
check("flat values correct", np.allclose(r.reshape(3,3).diagonal(), [0.04,0.01,0.09]))

print("\n" + ("ALL CHECKS PASSED" if ok else "SOMETHING FAILED"))
