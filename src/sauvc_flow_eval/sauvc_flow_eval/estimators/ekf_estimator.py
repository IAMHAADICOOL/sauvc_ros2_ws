#!/usr/bin/env python3
"""estimators/ekf_estimator.py — a small self-contained EKF for the comparison.

The request asked for a DEDICATED EKF inside this package rather than reusing the running
robot_localization node, so the comparison stands alone and can't be perturbed by the
live stack's tuning. This is a compact constant-velocity Kalman filter, NOT a
reimplementation of robot_localization — it exists to represent "the fused estimate" in
the comparison, fed by the same three sources the real EKF uses:

  state x = [px, py, pz, vx, vy]         (world frame = compare frame)
    px,py,pz : position (pz is depth as a world-Z in the compare frame)
    vx,vy    : world-frame planar velocity

  predict : constant-velocity model
  update  : - flow velocity (body -> world via yaw)  -> measures vx, vy
            - pressure depth (world Z)                -> measures pz
  yaw comes from the IMU orientation (already fused onboard), used only to rotate the
  flow measurement into the world; it is not part of the state.

Pure numpy, no ROS. Frame-agnostic: caller supplies measurements already in the compare
frame, so the same filter serves NED or ENU comparison.
"""

import numpy as np


class EkfEstimator:
    def __init__(self, q_pos=0.01, q_vel=0.5, r_flow=0.04, r_depth=4e-6):
        # state: px py pz vx vy
        self.x = np.zeros(5)
        self.P = np.eye(5) * 1.0
        self.q_pos = q_pos
        self.q_vel = q_vel
        self.R_flow = np.eye(2) * r_flow
        self.R_depth = r_depth
        self.t_prev = None
        self.initialized = False

    def _predict(self, dt):
        F = np.eye(5)
        F[0, 3] = dt          # px += vx dt
        F[1, 4] = dt          # py += vy dt
        self.x = F @ self.x
        Q = np.diag([self.q_pos, self.q_pos, self.q_pos,
                     self.q_vel, self.q_vel]) * dt
        self.P = F @ self.P @ F.T + Q

    def _step_time(self, t):
        if self.t_prev is None:
            self.t_prev = t
            return 0.0
        dt = t - self.t_prev
        self.t_prev = t
        if 0.0 < dt < 1.0:
            self._predict(dt)
        return dt

    def update_flow(self, vx_body, vy_body, yaw, t, r_var=None):
        """r_var: optional per-measurement variance (m/s)^2. IMPROVEMENT
        (quality-scaled noise): the caller can scale trust with the flow frame's
        quality (spread_px, n_inliers) like flow_velocity_node's variance model,
        instead of the fixed R_flow. None -> the constructor default."""
        self._step_time(t)
        c, s = np.cos(yaw), np.sin(yaw)
        # world velocity measurement from body velocity
        z = np.array([c * vx_body - s * vy_body,
                      s * vx_body + c * vy_body])
        H = np.zeros((2, 5)); H[0, 3] = 1.0; H[1, 4] = 1.0
        R = self.R_flow if r_var is None else np.eye(2) * max(float(r_var), 1e-6)
        self._kalman(z, H, R)
        self.initialized = True

    def update_position_xy(self, px, py, t, var_x, var_y):
        """MAP-BASED LANDMARK UPDATE. Apply a world-frame position pseudo-measurement
        derived from observing a mapped landmark:  p_meas = landmark_world - rel_obs_world.

        Per-axis variances make it ANISOTROPIC - the SAUVC gate's x is known from the
        rulebook but its y is randomized, so a gate observation constrains vehicle x
        tightly while y must stay untouched: pass var_y = 1e12 (a no-op on y). When a
        landmark's full (x, y) is known (e.g. stored at first sighting in the small-map
        scheme), pass both variances finite.

        var per axis = (landmark position variance) + (relative-observation variance).
        This is the editable seam for feature observations: the state stays
        [x y z vx vy] - pure localization against a known/stored map, NOT SLAM state
        augmentation."""
        self._step_time(t)
        z_meas = np.array([px, py])
        H = np.zeros((2, 5)); H[0, 0] = 1.0; H[1, 1] = 1.0
        R = np.diag([max(float(var_x), 1e-6), max(float(var_y), 1e-6)])
        self._kalman(z_meas, H, R)

    def update_depth(self, pz, t):
        self._step_time(t)
        z = np.array([pz])
        H = np.zeros((1, 5)); H[0, 2] = 1.0
        self._kalman(z, H, np.array([[self.R_depth]]))
        self.initialized = True

    def _kalman(self, z, H, R):
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(5) - K @ H) @ self.P

    @property
    def position(self):
        return self.x[0], self.x[1], self.x[2]

    @property
    def velocity(self):
        return self.x[3], self.x[4]
