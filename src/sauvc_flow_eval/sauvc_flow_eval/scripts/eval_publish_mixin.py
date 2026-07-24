#!/usr/bin/env python3
"""eval_publish_mixin.py — anchoring, publishing and the terminal report.

Split out of flow_eval_node.py unchanged. This is the output end of the node:
putting every track into one anchored coordinate system, publishing the
nav_msgs/Odometry, driving the live trajectory window, and printing the periodic
verification table. Mixed into FlowEvalNode, so every `self.` reference resolves
exactly as it did before the split.
"""
import math

import numpy as np

from nav_msgs.msg import Odometry

try:
    from cv_bridge import CvBridge
    import cv2
    _HAVE_CV = True
except Exception:
    _HAVE_CV = False

from sauvc_flow_eval.eval_common import gt_world_to_compare


class EvalPublishMixin:
    """_anchor / _publish / live plot timer / periodic print. Mixed into FlowEvalNode."""

    def _anchor(self, x, y, z):
        # add ground truth's start pose so every track begins where GT begins.
        # SCENE PARSER fallback: before the first GT message (or on hardware, where GT
        # never exists) use the start pose parsed from the .scn instead, converted to
        # the compare frame - so a changed scene spawn needs no code edit anywhere.
        a = self.gt_anchor
        if a is None and self.scene_start_ned is not None:
            a = tuple(gt_world_to_compare(np.array(self.scene_start_ned), self.frame))
        if a is None:
            return x, y, z
        return x + a[0], y + a[1], z + a[2]
    def _publish(self, key, x, y, z, stamp, vx=None, vy=None):
        self._latest[key] = (x, y, z, vx, vy)
        if self.traj is not None:
            self.traj.add(key, x, y)   # plotter ignores non-included keys
        self._maybe_print(stamp)
        od = Odometry()
        od.header.stamp = stamp
        od.header.frame_id = 'map_' + self.frame
        od.child_frame_id = 'base_link'
        od.pose.pose.position.x = float(x)
        od.pose.pose.position.y = float(y)
        od.pose.pose.position.z = float(z)
        od.twist.twist.linear.x = float(vx) if vx is not None else 0.0
        od.twist.twist.linear.y = float(vy) if vy is not None else 0.0
        self.pubs[key].publish(od)
    def _on_traj_timer(self):
        # Default single-threaded executor -> this runs in the same thread as
        # on_image's cv2.imshow calls, so mixing HighGUI here is safe.
        try:
            cv2.imshow('trajectory x-y', self.traj.render())
            cv2.waitKey(1)
        except Exception as e:
            # headless / no display: keep collecting points silently — the PNG
            # is still saved on shutdown, which is the part that matters.
            self.get_logger().warn(f'live_plot window unavailable ({e}) - '
                                   'will still save the PNG on exit',
                                   throttle_duration_sec=30.0)
    def _maybe_print(self, stamp):
        if not self.print_est:
            return
        t = stamp.sec + stamp.nanosec * 1e-9
        if t - self._last_print < self.print_period:
            return
        self._last_print = t
        def fmt(v):
            if v is None:
                v = (None,) * 5
            cells = [f"{c:+8.3f}" if c is not None else f"{'--':>8}" for c in v]
            return ' '.join(cells)
        order = ['ground_truth', 'dvl', 'flow', 'ekf', 'eskf', 'pressure',
                 'gtsam', 'tile_grid']
        lines = [f"\n─ estimates [{self.frame.upper()} frame]  "
                 f"x        y        z       vx       vy ──────"]
        for k in order:
            if k == 'gtsam' and not self.gtsam.available:
                continue
            lines.append(f"  {k:<13} {fmt(self._latest[k])}")
        # VERIFICATION LINE (checks the claim in gtsam_estimator.current_ned_yaw's
        # docstring): if this graph really is immune to the .scn's yaw_drift because it
        # integrates raw gyro upstream of where Stonefish injects that drift, gtsam_yaw
        # should track gt_yaw far more closely than imu_yaw does, and the gap between
        # imu_yaw and gt_yaw should be the one that keeps growing over a long run.
        if self.gtsam.available and self.gtsam.initialized and self.gt_yaw_ned is not None:
            gy = self.gtsam.current_ned_yaw()
            if gy is not None:
                d = lambda a, b: math.degrees(math.atan2(math.sin(a - b), math.cos(a - b)))
                lines.append(
                    f"  yaw[deg]      gt={math.degrees(self.gt_yaw_ned):+7.2f}  "
                    f"imu={math.degrees(self.yaw_ned):+7.2f} (err {d(self.yaw_ned, self.gt_yaw_ned):+6.2f})  "
                    f"gtsam={math.degrees(gy):+7.2f} (err {d(gy, self.gt_yaw_ned):+6.2f})")
            # GYRO-BIAS ESTIMATE: with the patched Stonefish IMU, yaw drift enters
            # as a real gyro z-bias (= the .scn yaw_drift, rad/s). GTSAM's bias state
            # should converge to it. Printing gz here (rad/s AND deg/min for eyeball
            # comparison to the .scn yaw_drift) shows whether the graph is ESTIMATING
            # the bias, not merely suffering the drift. Expected: gz -> ~yaw_drift.
            gb = self.gtsam.gyro_bias()
            if gb is not None:
                lines.append(
                    f"  gyro_bias[rad/s] bx={gb[0]:+.5f} by={gb[1]:+.5f} "
                    f"bz={gb[2]:+.5f}  (bz={math.degrees(gb[2])*60:+.2f} deg/min "
                    f"-> compare to .scn yaw_drift)")
        # YAW UPGRADE (2026-07-23): the 7-state EKF's own psi/bias, alongside gt/
        # imu/gtsam/lane in one row, plus both bias estimates in deg/min for the
        # falsifiable convergence check (see _yaw_report_line's docstring on the
        # node). Appended to `lines` rather than print()'d separately so it stays
        # inside this function's single buffered block like everything else here.
        lines.append(self._yaw_report_line())
        # LANE-HEADING VISIBILITY: previously this print's imu= column was the ONLY
        # yaw info shown, and it's raw self.yaw_ned regardless of whether lane fusion
        # ran — there was no way to tell from these logs whether lane_heading_node was
        # doing anything at all. This line shows the ACTUAL yaw fed to EKF/flow/GTSAM
        # this cycle (self.yaw_ned_fused), whether lane fusion was active THIS cycle,
        # and — when GT is available (sim) — whether that fused yaw is closer to truth
        # than raw IMU would have been, i.e. whether lane fusion is actually helping.
        if self.use_lane:
            d = lambda a, b: math.degrees(math.atan2(math.sin(a - b), math.cos(a - b)))
            tag = 'ACTIVE' if self.lane_active else 'inactive(IMU fallback)'
            extra = ''
            if self.gt_yaw_ned is not None:
                e_fused = d(self.yaw_ned_fused, self.gt_yaw_ned)
                e_imu = d(self.yaw_ned, self.gt_yaw_ned)
                extra = (f'  fused_err={e_fused:+6.2f}  imu_err={e_imu:+6.2f}  '
                        f'{"better" if abs(e_fused) < abs(e_imu) else "WORSE/no-help"}')
            lines.append(f"  lane[{tag:<22}] raw_lane_yaw="
                        + (f"{math.degrees(self.lane_yaw):+7.2f}deg"
                           if self.lane_yaw is not None else "   never rx'd")
                        + extra)
        # UPGRADE H: the SLAM map, in WORLD NED so the numbers compare to the scene
        # file (and to landmark_truth_node's parse) directly — anchor added back.
        # Shows the EKF's mean±sigma per feature, chi2 rejection counts, and the
        # GTSAM graph's landmark estimates when any have been born.
        if self.lm_mode == 'slam' and self.ekf.features:
            ax = self._anchor_x_ned()
            ay = self.gt_anchor[1] if self.gt_anchor is not None else (
                self.scene_start_ned[1] if self.scene_start_ned is not None else None)
            if ax is not None and ay is not None:
                lines.append('  ekf map (world NED):')
                for n in self.ekf.feature_names():
                    fx0, fy0, vx0, vy0 = self.ekf.feature_estimate(n)
                    rej = self.ekf.rejected.get(n, 0)
                    lines.append(
                        f"    {n:<16} x={fx0 + ax:+7.2f}±{math.sqrt(vx0):4.2f} "
                        f"y={fy0 + ay:+7.2f}±{math.sqrt(vy0):4.2f}"
                        + (f'  [chi2-rejected {rej}]' if rej else ''))
                lm_est = self.gtsam.landmark_estimates() if self.gtsam.available else {}
                for n in sorted(lm_est):
                    e = lm_est[n]
                    lines.append(f"    {n:<16} x={e[0] + ax:+7.2f} "
                                 f"y={e[1] + ay:+7.2f}  [gtsam]")
        print('\n'.join(lines))
