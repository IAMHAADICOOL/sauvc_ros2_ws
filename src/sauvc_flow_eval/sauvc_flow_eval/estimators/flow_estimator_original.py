#!/usr/bin/env python3
"""estimators/flow_estimator.py — flow + altitude metric velocity, and its visualization.

Wraps the EXISTING sauvc_localization.flow_core.FlowVelocityEstimator unchanged. That
estimator already produces METRIC body velocity by scaling pixel flow with altitude
(pool_depth - depth). There is no scale ambiguity to resolve here — pressure supplies the
range. This module adds:

  * a thin adapter that feeds gray frame + gyro + altitude and returns body (vx, vy)
  * an optical-flow OVERLAY image (tracked features + median flow vector) for the OpenCV
    window the request asked for

Kept deliberately separate from the ROS node so the node just calls estimate()/overlay().
"""

import numpy as np
import cv2

from sauvc_localization.flow_core import FlowVelocityEstimator


class FlowEstimator:
    def __init__(self, fx, fy, cx, cy, **kw):
        self.core = FlowVelocityEstimator(fx, fy, cx, cy, **kw)
        self.last = None       # last result dict, for the overlay
        self.last_gray = None

    def estimate(self, gray, dt, gyro_xy_cam, altitude):
        """Returns dict(vx, vy, ...) in body frame, or None."""
        res = self.core.process(gray, dt, gyro_xy_cam, altitude)
        if res is None:
            self.dropouts = getattr(self, 'dropouts', 0) + 1
        self.last = res
        self.last_gray = gray
        return res

    def overlay(self, frame_bgr):
        """Draw tracked corners and the median flow vector onto a BGR frame."""
        img = frame_bgr.copy()
        pts = getattr(self.core, 'prev_pts', None)
        if pts is not None:
            for p in pts.reshape(-1, 2):
                cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
        if self.last and 'flow_px' in self.last:
            h, w = img.shape[:2]
            cx, cy = w // 2, h // 2
            du, dv = self.last['flow_px']

            # TWO arrows, because "the arrow points opposite my motion" is EXPECTED, not a
            # bug, and showing only one invites that confusion:
            #
            #   RED  = raw pixel flow (du, dv). Ground features stream OPPOSITE to travel,
            #          exactly like scenery past a train window. Forward motion -> this
            #          points backward. That is correct.
            #   CYAN = recovered velocity direction (image-plane), which is -flow. This
            #          points ALONG your travel. This is what becomes /eval/flow vx,vy.
            #
            # If CYAN disagrees with your actual motion, THEN the sign convention is wrong
            # (fix sign_x/sign_y/swap_xy in flow_sim.yaml) — verify against the terminal
            # print of /eval/ground_truth vs /eval/flow, not by eyeballing the arrow.
            k = 8.0
            cv2.arrowedLine(img, (cx, cy), (int(cx + du * k), int(cy + dv * k)),
                            (0, 0, 255), 2, tipLength=0.3)          # raw flow (red)
            cv2.arrowedLine(img, (cx, cy), (int(cx - du * k), int(cy - dv * k)),
                            (255, 255, 0), 2, tipLength=0.3)        # velocity dir (cyan)

            cv2.putText(img, "red=pixel flow (opposes motion)  cyan=velocity (along motion)",
                        (8, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            txt = (f"n={self.last['n_inliers']}/{self.last['n_tracked']} "
                   f"v=({self.last['vx']:+.2f},{self.last['vy']:+.2f}) m/s")
            cv2.putText(img, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1, cv2.LINE_AA)
        return img
