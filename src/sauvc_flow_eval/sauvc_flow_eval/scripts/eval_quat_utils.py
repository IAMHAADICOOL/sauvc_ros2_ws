#!/usr/bin/env python3
"""eval_quat_utils.py — quaternion / frame conversion helpers for flow_eval_node.

Split out of flow_eval_node.py unchanged. These are the small, pure functions the
node uses to move between the ENU/FLU quaternions the shims publish and the
NED/FRD conventions every estimator runs in. Each carries its own verification
note in its docstring — those notes are the record of what was actually checked
against gtsam.Rot3, so they travel with the code.
"""
import math

import numpy as np

from sauvc_sim_bridge.frames import flu_frd_to_ned_wxyz


def _enu_quat_to_ned_wxyz(x, y, z, w):
    return flu_frd_to_ned_wxyz(x, y, z, w)
def _quat_to_R(x, y, z, w):
    n = np.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]])
def _yaw_from_quat_xyzw(x, y, z, w):
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
def _rp_from_quat_wxyz(w, x, y, z):
    """Extract (roll, pitch) only from a NED/FRD wxyz quaternion — standard aerospace
    ZYX convention. Verified by round-trip against real gtsam.Rot3.Ypr/toQuaternion
    (exact match to 1e-6) before use in the GTSAM attitude-prior fix below."""
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(sinp)
    return roll, pitch
def _quat_wxyz_from_rpy(roll, pitch, yaw):
    """Inverse of the above with an arbitrary yaw substituted in. Verified against
    gtsam.Rot3.Ypr(yaw, pitch, roll).toQuaternion() — exact match to 1e-6."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return (w, x, y, z)
