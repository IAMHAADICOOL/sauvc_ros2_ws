"""
frames.py — the ONLY place in the workspace where NED/FRD becomes ENU/FLU.

Why this file exists
--------------------
Stonefish is NED throughout. Verified against the upstream source, not the docs:
`ROSInterface::PublishOdometry` hard-codes `msg.header.frame_id = "world_ned"`, and
nothing in the ROS wrapper converts anything. So every topic under the simulator's
robot namespace (`/sauvc_auv/*`) carries:

    world frame : NED   (x North, y East, z Down)
    body  frame : FRD   (x Forward, y Right, z Down)

Your stack is REP-103 throughout:

    world frame : ENU   (x East, y North, z Up)
    body  frame : FLU   (x Forward, y Left, z Up)

The rule this module enforces: `/sauvc_auv/*` is NED and belongs to the simulator.
Everything on the un-namespaced topics (`/imu/data`, `/depth`, `/flow/twist`,
`/odometry/filtered`) is ENU and belongs to your stack. The shim nodes are the only
code allowed to see both, and they must get their conversion from here. If a sign
flip ever appears anywhere else in the workspace, it is a bug — fix it here instead.

The conversion is the standard MAVROS `ftf` pair of fixed 180-degree rotations:

    world NED -> ENU : 180 deg about the axis (sqrt(2)/2, sqrt(2)/2, 0)
                       as a vector map: (x, y, z) -> (y, x, -z)
    body  FRD -> FLU : 180 deg about the x axis
                       as a vector map: (x, y, z) -> (x, -y, -z)

Both are involutions (each is its own inverse), which is why the same functions run
in both directions and why there is no `enu_to_ned` twin below.

Orientation composes as:

    q_enu_flu = Q_NED_ENU  *  q_ned_frd  *  Q_FRD_FLU

Quaternions here are (x, y, z, w) — ROS message order, NOT scipy's or Eigen's
constructor order. Getting this wrong is silent and produces a plausible-looking
wrong attitude, so everything below is expressed in one convention only.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Fixed rotations, as (x, y, z, w).
# ---------------------------------------------------------------------------

_S = np.sqrt(2.0) / 2.0

#: World rotation: 180 deg about (sqrt(2)/2, sqrt(2)/2, 0). Equivalent to
#: rpy(pi, 0, pi/2) in ZYX. Maps NED -> ENU and ENU -> NED.
Q_NED_ENU = np.array([_S, _S, 0.0, 0.0])

#: Body rotation: 180 deg about x. Maps FRD -> FLU and FLU -> FRD.
Q_FRD_FLU = np.array([1.0, 0.0, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Quaternion algebra (x, y, z, w)
# ---------------------------------------------------------------------------

def quat_mul(q1, q2):
    """Hamilton product of two (x, y, z, w) quaternions."""
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    return np.array([
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
    ])


def quat_normalize(q):
    """Normalize, guarding against the zero quaternion."""
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    return q / n


# ---------------------------------------------------------------------------
# Vector conversions
# ---------------------------------------------------------------------------

def ned_to_enu_vec(v):
    """World-frame vector NED -> ENU: (x, y, z) -> (y, x, -z). Self-inverse."""
    x, y, z = v
    return np.array([y, x, -z])


def frd_to_flu_vec(v):
    """Body-frame vector FRD -> FLU: (x, y, z) -> (x, -y, -z). Self-inverse.

    Use this for anything measured in the body: linear acceleration, angular
    velocity, DVL velocity, optical-flow velocity.
    """
    x, y, z = v
    return np.array([x, -y, -z])


# ---------------------------------------------------------------------------
# Orientation conversion
# ---------------------------------------------------------------------------

def ned_frd_quat_to_enu_flu(q):
    """Convert an attitude quaternion from (NED world, FRD body) to (ENU, FLU).

    `q` is the rotation taking body FRD into world NED, as Stonefish's IMU and
    odometry publish it. Returns the rotation taking body FLU into world ENU,
    as robot_localization expects it. (x, y, z, w) in, (x, y, z, w) out.
    """
    return quat_normalize(quat_mul(quat_mul(Q_NED_ENU, quat_normalize(q)), Q_FRD_FLU))


# ---------------------------------------------------------------------------
# Covariance conversion
# ---------------------------------------------------------------------------

def rot_matrix_ned_enu():
    """3x3 world NED->ENU rotation as a matrix. Self-inverse, orthogonal."""
    return np.array([[0.0, 1.0, 0.0],
                     [1.0, 0.0, 0.0],
                     [0.0, 0.0, -1.0]])


def rot_matrix_frd_flu():
    """3x3 body FRD->FLU rotation as a matrix. Self-inverse, orthogonal."""
    return np.diag([1.0, -1.0, -1.0])


def rotate_cov3(cov, R):
    """Conjugate a 3x3 covariance through rotation R: Sigma' = R Sigma R^T.

    Both rotations in this module are signed axis permutations, so for a
    DIAGONAL covariance this reduces to permuting the diagonal — the sign flips
    square away and never survive. That is worth knowing, because it means a
    sign error in a diagonal covariance is invisible: it will not blow up, it
    will just be silently attached to the wrong axis. Off-diagonal terms do
    change sign, so use the full conjugation rather than hand-permuting.

    `cov` is 3x3 or a flat 9-vector (ROS row-major); returns the same shape.
    """
    flat = np.asarray(cov, dtype=float).size == 9 and np.asarray(cov).ndim == 1
    C = np.asarray(cov, dtype=float).reshape(3, 3)
    out = R @ C @ R.T
    return out.reshape(9) if flat else out


def cov3_ned_to_enu(cov):
    """World-frame 3x3 (or flat-9) covariance, NED -> ENU."""
    return rotate_cov3(cov, rot_matrix_ned_enu())


def cov3_frd_to_flu(cov):
    """Body-frame 3x3 (or flat-9) covariance, FRD -> FLU."""
    return rotate_cov3(cov, rot_matrix_frd_flu())


def enu_flu_quat_to_ned_frd(q):
    """Inverse of ned_frd_quat_to_enu_flu: (ENU world, FLU body) attitude -> (NED, FRD).

    Both Q_NED_ENU and Q_FRD_FLU are involutions (self-inverse), and the forward map is
    q_enu = Q_NED_ENU * q_ned * Q_FRD_FLU. Inverting:
        q_ned = Q_NED_ENU * q_enu * Q_FRD_FLU
    i.e. structurally identical, because each fixed rotation equals its own inverse.
    Input/output (x, y, z, w).
    """
    return quat_normalize(quat_mul(quat_mul(Q_NED_ENU, quat_normalize(q)), Q_FRD_FLU))


def flu_frd_to_ned_wxyz(x, y, z, w):
    """Convenience: ENU/FLU quaternion (x,y,z,w) -> NED/FRD quaternion (w,x,y,z),
    the order gtsam.Rot3.Quaternion expects."""
    qx, qy, qz, qw = enu_flu_quat_to_ned_frd([x, y, z, w])
    return (qw, qx, qy, qz)
