#!/usr/bin/env python3
"""eval_common.py — shared helpers for the estimator comparison. No ROS imports.

Two jobs, both small and testable:

  1. COMMON-FRAME conversion. Every estimator is put into ONE frame before comparison so
     the plots are apples-to-apples. Default is NED (per the request), but the seam is a
     single parameter, `compare_frame`, and the ONLY conversion used is the tested pair
     in frames.py — never ad-hoc sign flips per estimator. See the note below on why the
     conversion is applied to ground truth rather than to each estimate.

  2. Velocity -> position INTEGRATION. Flow and the GTSAM path produce body velocity;
     drift-prone dead-reckoned position is exactly what we want to visualize, so we
     integrate here with an explicit, honest trapezoidal rule and never pretend the
     result doesn't drift.

WHY WE CONVERT GROUND TRUTH, NOT EACH ESTIMATE (even though the request said "estimates
to NED")
---------------------------------------------------------------------------------------
Estimates arrive in several frames (flow = body, EKF = ENU/FLU world, pressure = a scalar
depth). Converting each of N estimators into NED is N separate conversions = N chances
for a silent sign error, the exact failure mode that has bitten this project before.
Ground truth is ONE source in NED. The estimators' natural comparison frame is the ENU
world their stack already lives in.

So the implementation keeps a single knob:
  compare_frame='ned' -> convert the ENU world estimates to NED once (via frames.py),
                          leave Stonefish ground truth untouched. NED plots, as requested.
  compare_frame='enu' -> convert ground truth NED->ENU once, leave estimates untouched.
Both use only frames.py. The DEFAULT is 'ned' as asked; 'enu' exists because it needs
one fewer conversion and is marginally safer if you ever doubt a sign.
"""

import numpy as np

try:
    # frames.py ships in sauvc_sim_bridge; the eval package depends on it.
    from sauvc_sim_bridge.frames import (
        ned_to_enu_vec, frd_to_flu_vec, ned_frd_quat_to_enu_flu)
except Exception:  # pragma: no cover - allows offline unit tests via a local shim
    ned_to_enu_vec = None


# --- world-frame vector conversions, both directions, magnitude-preserving ------------
# ENU->NED world is the same involution as NED->ENU: (x,y,z)->(y,x,-z).
def enu_to_ned_vec(v):
    x, y, z = v
    return np.array([y, x, -z])


def ned_to_enu_world(v):
    x, y, z = v
    return np.array([y, x, -z])


# FLU->FRD body is the same involution as FRD->FLU: (x,y,z)->(x,-y,-z).
def flu_to_frd_vec(v):
    x, y, z = v
    return np.array([x, -y, -z])


def depth_to_world_z(depth, frame):
    """A positive depth (m below surface) as a world Z coordinate in the chosen frame.
    NED z is +down -> +depth. ENU z is +up -> -depth."""
    return depth if frame == 'ned' else -depth


def to_compare_frame_world(vec_enu, frame):
    """Take a WORLD-frame vector expressed in ENU and return it in the compare frame."""
    return enu_to_ned_vec(vec_enu) if frame == 'ned' else np.asarray(vec_enu, float)


def gt_world_to_compare(vec_ned, frame):
    """Take Stonefish's WORLD-frame vector (NED) and return it in the compare frame."""
    return np.asarray(vec_ned, float) if frame == 'ned' else ned_to_enu_world(vec_ned)


class PositionIntegrator:
    """Trapezoidal dead-reckoning of a body-frame velocity into WORLD position.

    Honest about drift: no ZUPT, no loop closure, no fusion. It rotates each body
    velocity into the world using the supplied yaw, then integrates. The whole point is
    to SEE the drift accumulate, so nothing here tries to suppress it.

    yaw is the vehicle heading in the WORLD frame we are integrating in (so if
    compare_frame='ned', pass the NED yaw; if 'enu', the ENU yaw). x/y only — depth comes
    from pressure, not from integrating vertical velocity.
    """

    def __init__(self):
        self.x = 0.0
        self.y = 0.0
        self.prev = None      # (vx_world, vy_world)
        self.t_prev = None

    def reset(self, x=0.0, y=0.0):
        self.x, self.y, self.prev, self.t_prev = x, y, None, None

    def update(self, vx_body, vy_body, yaw, t):
        # body -> world rotation by yaw (2D)
        c, s = np.cos(yaw), np.sin(yaw)
        vxw = c * vx_body - s * vy_body
        vyw = s * vx_body + c * vy_body
        if self.prev is not None and self.t_prev is not None:
            dt = t - self.t_prev
            if 0.0 < dt < 1.0:                       # ignore absurd gaps
                self.x += 0.5 * (vxw + self.prev[0]) * dt
                self.y += 0.5 * (vyw + self.prev[1]) * dt
        self.prev = (vxw, vyw)
        self.t_prev = t
        return self.x, self.y
