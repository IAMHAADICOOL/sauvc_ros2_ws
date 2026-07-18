#!/usr/bin/env python3
"""
flow_core.py — Downward-camera optical-flow velocity ("DIY DVL") for SAUVC.

Method:
  1. Track sparse corners frame-to-frame with pyramidal Lucas-Kanade (no loop closure,
     so repetitive pool tiles are NOT a problem).
  2. Take the MEDIAN pixel displacement (robust to caustics / moving highlights) and
     gate outliers with MAD (median absolute deviation).
  3. Derotate: subtract the apparent flow caused by pitch/roll rates (gyro), so tilting
     the vehicle is not mistaken for translation.
  4. Scale by altitude above the pool floor (from pressure sensor: pool_depth - depth)
     to get metric velocity, and rotate into the body frame.

Publishes: geometry_msgs/TwistWithCovarianceStamped on /flow/twist
Subscribes: sensor_msgs/Image on /camera_down/image_raw
            sensor_msgs/Imu   on /imu/data        (angular velocity, body frame)
            std_msgs/Float32  on /altitude        (meters above pool floor)

The math lives in FlowVelocityEstimator (no ROS imports) so you can unit-test it and run
it offline with offline_flow_test.py.
"""

import math
import numpy as np
import cv2


class FlowVelocityEstimator:
    """Pure-python core. Feed grayscale frames + gyro rates + altitude, get body velocity."""

    def __init__(self, fx, fy, cx, cy,
                 max_corners=150, quality=0.01, min_distance=12,
                 swap_xy=False, sign_x=1.0, sign_y=1.0,
                 min_features=25):
        # Camera intrinsics — MUST be from an UNDERWATER calibration.
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.max_corners = max_corners
        self.quality = quality
        self.min_distance = min_distance
        self.min_features = min_features
        # Mounting convention fix-ups (set from the hand-push test, see README Phase 3).
        self.swap_xy = swap_xy
        self.sign_x = sign_x
        self.sign_y = sign_y

        self.prev_gray = None
        self.prev_pts = None
        self.lk_params = dict(winSize=(21, 21), maxLevel=3,
                              criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                                        30, 0.01))

    def _detect(self, gray):
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_corners,
                                      qualityLevel=self.quality,
                                      minDistance=self.min_distance)
        return pts  # shape (N,1,2) float32 or None

    def process(self, gray, dt, gyro_xy_cam, altitude):
        """
        gray        : uint8 grayscale frame (undistorted, or low-distortion lens)
        dt          : seconds since previous frame
        gyro_xy_cam : (wx, wy) angular rates in the CAMERA frame [rad/s]
                      (camera x = image u/right, camera y = image v/down)
        altitude    : meters between camera and pool floor
        Returns dict with vx, vy (body frame, m/s), quality info; or None if no estimate.
        """
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None or self.prev_pts is None or len(self.prev_pts) < self.min_features:
            self.prev_gray = gray
            self.prev_pts = self._detect(gray)
            return None

        nxt, status, _err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray,
                                                     self.prev_pts, None, **self.lk_params)
        if nxt is None:
            self.prev_gray, self.prev_pts = gray, self._detect(gray)
            return None

        good = status.reshape(-1) == 1
        p0 = self.prev_pts.reshape(-1, 2)[good]
        p1 = nxt.reshape(-1, 2)[good]
        n_tracked = len(p0)

        # Re-seed for next iteration regardless of what happens below.
        self.prev_gray = gray
        self.prev_pts = self._detect(gray)

        if n_tracked < self.min_features or dt <= 0 or altitude is None or altitude < 0.1:
            return None

        d = p1 - p0                                   # per-feature pixel displacement
        med = np.median(d, axis=0)                    # robust central flow (du, dv)
        mad = np.median(np.abs(d - med), axis=0) + 1e-6
        # Keep features within 4 MAD of the median (rejects caustic sparkles, fish, ropes).
        inlier = np.all(np.abs(d - med) < 4.0 * mad + 1.0, axis=1)
        n_inliers = int(inlier.sum())
        if n_inliers < self.min_features:
            return None
        du, dv = np.median(d[inlier], axis=0)         # pixels per frame

        # --- Derotation ---------------------------------------------------------------
        # Near the image center, rotational flow:  u_dot ~= -f * wy ,  v_dot ~= +f * wx
        # Subtract it to isolate translational flow.
        wx, wy = gyro_xy_cam
        du_t = du - (-self.fx * wy * dt)
        dv_t = dv - (+self.fy * wx * dt)

        # --- Scale to metric camera-frame velocity --------------------------------------
        # Ground point image motion is opposite camera translation: u_dot = -f * Vx_c / h
        vx_cam = -(du_t / dt) * altitude / self.fx    # camera x = image right
        vy_cam = -(dv_t / dt) * altitude / self.fy    # camera y = image down

        # --- Camera -> body mapping (fix with hand-push test) ---------------------------
        # Default assumption: image "up" (-v) is vehicle forward (+x body),
        # image "right" (+u) is vehicle right (-y body in ROS, since y is left).
        bx, by = -vy_cam, -vx_cam
        if self.swap_xy:
            bx, by = by, bx
        bx *= self.sign_x
        by *= self.sign_y

        # Quality: spread of inlier flow (pixels) — big spread = unreliable frame.
        spread = float(np.mean(np.std(d[inlier], axis=0)))
        return dict(vx=float(bx), vy=float(by),
                    n_tracked=n_tracked, n_inliers=n_inliers,
                    spread_px=spread, flow_px=(float(du), float(dv)))


