#!/usr/bin/env python3
"""eval_helpers_mixin.py — altitude, gyro-sync and yaw helpers for flow_eval_node.

Split out of flow_eval_node.py unchanged. These five methods are the "upgrades"
group: they all turn raw incoming sensor state into the corrected quantities the
estimators actually consume (camera altitude, time-aligned gyro, the flow->IMU yaw
extrinsic, and the lane-heading-fused yaw). Mixed into FlowEvalNode, so every
`self.` reference resolves exactly as it did before the split.
"""
import math

import numpy as np


class EvalHelpersMixin:
    """UPGRADE A/B/C/D helpers + the flow->IMU yaw autocal. Mixed into FlowEvalNode."""

    # ---- helpers (upgrades) ----
    def _floor_depth_at(self, x_ned):
        if not self.use_profile:
            return float(self.get_parameter('pool_depth').value)
        xp = (0.0 if x_ned is None else x_ned) + self.prof_off
        return float(np.interp(xp, self.prof_x, self.prof_d))
    def _camera_altitude(self, t):
        """UPGRADE A+B: camera-to-floor range along the optical axis, computed
        entirely from data this node already has. Independent of depth_shim, so the
        odom-feed launch trap that broke three runs cannot recur here."""
        cam_depth = self.depth + self.sensor_above + self.cam_below
        # alt = max(self._floor_depth_at(self.gt_x_ned) - cam_depth, 0.05)
        alt = max(self._floor_depth_at(self._alt_profile_x()) - cam_depth, 0.05)
        if self.tilt_comp:
            c = math.cos(self.roll) * math.cos(self.pitch)
            alt = alt / max(c, 0.5)      # clamp: >60 deg tilt would blow up the scale
        # cross-check the shim's /altitude and diagnose its misconfiguration loudly
        if (self.altitude is not None and self.alt_warn_m > 0.0
                and abs(self.altitude - alt) > self.alt_warn_m
                and t - self._alt_warned_t > 10.0):
            self._alt_warned_t = t
            self.get_logger().warn(
                f'/altitude from depth_shim ({self.altitude:.2f} m) disagrees with the '
                f'in-node camera altitude ({alt:.2f} m) by more than {self.alt_warn_m} m.'
                ' The shim is almost certainly still evaluating the floor profile at a '
                'stale x (its odom_topic feed is not connected). This node is UNAFFECTED'
                ' (self_altitude is on), but fix the shim before hardware parity work.')
        return alt
    def _gyro_at(self, t_query):
        """UPGRADE C: gyro interpolated at t_query from the ring buffer; falls back
        to the latest sample when the buffer cannot bracket the query."""
        if len(self._gyro_buf) < 2:
            return self.gyro_body
        buf = list(self._gyro_buf)
        ts = np.array([b[0] for b in buf])
        if not (ts[0] <= t_query <= ts[-1]):
            return buf[-1][1:4]
        wx = float(np.interp(t_query, ts, [b[1] for b in buf]))
        wy = float(np.interp(t_query, ts, [b[2] for b in buf]))
        wz = float(np.interp(t_query, ts, [b[3] for b in buf]))
        return (wx, wy, wz)
    def _update_yaw_autocal(self, vx_b, vy_b, yaw_raw, t):
        """Track the flow-body -> IMU-yaw offset from ground truth on straight legs.
        EWMA FIX (replaces freeze-after-N): the dominant contributor is the .scn's
        yaw_drift — a GROWING random walk on the published IMU yaw, not a fixed
        extrinsic. A frozen estimate is exact at the freeze instant and accumulates
        error at the full drift rate forever after; a never-freezing EWMA on the unit
        circle instead tracks it with a small BOUNDED lag ~= drift_per_sample/alpha
        (~0.03 deg at the current .scn rate). Rotates the RAW flow body velocity into
        the compare-frame world with yaw_raw and measures the angle to the GT
        world-velocity direction; sim-only (needs GT), harmless on hardware."""
        gv = self._gt_vel
        if gv is None:
            return
        gs = math.hypot(gv[0], gv[1])
        c, s = math.cos(yaw_raw), math.sin(yaw_raw)
        fwx = c * vx_b - s * vy_b            # raw flow world velocity (compare frame)
        fwy = s * vx_b + c * vy_b
        fs = math.hypot(fwx, fwy)
        if gs < self.yawcal_min_speed or fs < self.yawcal_min_speed:
            return
        # offset delta such that rotating flow by (yaw + delta) aligns it with GT
        d = math.atan2(gv[1], gv[0]) - math.atan2(fwy, fwx)
        a = self.yawcal_alpha
        if self._yawcal_n == 0:
            self._yawcal_s, self._yawcal_c = math.sin(d), math.cos(d)
        else:
            self._yawcal_s = (1.0 - a) * self._yawcal_s + a * math.sin(d)
            self._yawcal_c = (1.0 - a) * self._yawcal_c + a * math.cos(d)
        self._yawcal_n += 1
        if self._yawcal_n >= self.yawcal_min_n:
            self.flow_yaw_offset = math.atan2(self._yawcal_s, self._yawcal_c)
        if t - self._yawcal_log_t > 30.0 and self._yawcal_n >= self.yawcal_min_n:
            self._yawcal_log_t = t
            self.get_logger().info(
                'flow->IMU yaw offset (EWMA, tracking — never frozen): %+.2f deg '
                'over %d samples. Growing = the .scn yaw_drift accumulating, by '
                'design. Hardware: measure once, set -p flow_yaw_offset and '
                '-p flow_yaw_autocal:=false.'
                % (math.degrees(self.flow_yaw_offset), self._yawcal_n))
    def _fused_yaws(self, t_wall):
        """UPGRADE D: (yaw_compare, yaw_ned) with fresh lane-heading substitution.
        Also records, on self, WHETHER fusion applied this cycle and what the
        resulting yaw actually is — this is what makes lane_heading's effect
        (or lack of it) visible in the periodic print and in the summary log
        below, instead of the print always showing raw IMU yaw regardless of
        whether lane fusion ran at all."""
        yaw_ned = self.yaw_ned
        self.lane_active = False
        if not self.use_lane:
            pass  # feature off; nothing to report
        elif self.lane_yaw is None or t_wall - self.lane_yaw_t >= self.lane_fresh_s:
            self._lane_reject_stale_n += 1
        else:
            # FIX(ENU->NED is a REFLECTION, not an offset): lane_heading_node
            # publishes an ENU-convention heading (0=East, CCW-positive) minus its
            # OWN pool_axis_offset param — run that node with pool_axis_offset LEFT
            # AT ITS DEFAULT 0.0 for this fusion (it exists for a different,
            # mission-relative purpose; see that node's docstring). NED yaw
            # (bearing: 0=North, CW-positive) is yaw_ned = pi/2 - yaw_enu — verified
            # both against eval_common's vector convention (NED(x,y,z)=ENU(y,x,-z))
            # and against compass bearings at all four cardinal headings.
            # The OLD formula here (cand = lane_yaw + pool_axis_ned_yaw) assumed a
            # constant SHIFT between the two conventions, which is only ever exact
            # at the single heading it happened to be checked against (spawn) and
            # is off by up to a full 180 deg elsewhere — verified numerically
            # (error = -2*yaw_ned, mod wrap). In practice the sanity gate below was
            # silently REJECTING the lane fusion on every real turn and falling
            # back to IMU-only, so lane heading was never actually engaging except
            # near the start heading. pool_axis_ned_yaw is now a small NED-frame
            # TRIM applied AFTER the reflection (default 0; use it only for a
            # genuine residual, e.g. floor tiles not quite aligned with true pool
            # axis at the venue — not for ENU/NED conversion, which is now exact).
            cand = (np.pi / 2.0 - self.lane_yaw) + self.pool_axis_ned
            d = (cand - self.yaw_ned + np.pi) % (2 * np.pi) - np.pi
            if abs(d) > np.radians(30.0):
                self._lane_reject_gate_n += 1
                if t_wall - self._lane_warned > 5.0:
                    self._lane_warned = t_wall
                    self.get_logger().warn(
                        f'lane yaw rejected: disagrees with IMU-NED yaw by '
                        f'{np.degrees(d):+.0f} deg — run lane_heading_node with '
                        '-p pool_axis_offset:=0.0 (its default; the ENU->NED '
                        'reflection and any trim are handled here). Falling back '
                        'to IMU yaw.')
            else:
                yaw_ned = cand
                self.lane_active = True
                self._lane_accept_n += 1
        self.yaw_ned_fused = yaw_ned
        if self.use_lane and t_wall - self._lane_summary_last_t > self._lane_summary_period:
            self._lane_summary_last_t = t_wall
            tot = self._lane_accept_n + self._lane_reject_stale_n + self._lane_reject_gate_n
            if tot > 0:
                d_now = None
                if self.gt_yaw_ned is not None:
                    d_now = math.degrees(math.atan2(math.sin(yaw_ned - self.gt_yaw_ned),
                                                    math.cos(yaw_ned - self.gt_yaw_ned)))
                d_imu = None
                if self.gt_yaw_ned is not None:
                    d_imu = math.degrees(math.atan2(math.sin(self.yaw_ned - self.gt_yaw_ned),
                                                    math.cos(self.yaw_ned - self.gt_yaw_ned)))
                self.get_logger().info(
                    f'lane_heading effectiveness: accepted {self._lane_accept_n}/{tot} '
                    f'({100.0*self._lane_accept_n/tot:.0f}%), rejected-stale '
                    f'{self._lane_reject_stale_n} (no fresh /heading/pool_relative), '
                    f'rejected-gate {self._lane_reject_gate_n} (disagreed >30 deg). '
                    + (f'Right now: fused_yaw err={d_now:+.2f} deg vs raw IMU err='
                       f'{d_imu:+.2f} deg — {"HELPING" if abs(d_now) < abs(d_imu) else "NOT currently helping"}.'
                       if d_now is not None else
                       'No GT yaw yet to compare against (sim-only check).'))
        yaw_cmp = yaw_ned if self.frame == 'ned' else (np.pi / 2 - yaw_ned)
        return yaw_cmp, yaw_ned
