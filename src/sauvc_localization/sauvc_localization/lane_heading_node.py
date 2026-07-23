#!/usr/bin/env python3
"""
lane_heading_node.py — Phase 4. Absolute heading from pool floor lines (mod 90°).

Idea: competition pools have lane lines / tile grout aligned with the pool axes. The
dominant line direction in the DOWN camera gives vehicle yaw relative to the pool,
modulo 90°. Fused as a slow complementary correction to the gyro-integrated yaw, this
cancels drift with no magnetometer. The mod-90 ambiguity is fine because gyro yaw never
drifts ~45° between corrections.

Subscribes: /camera_down/image_raw (sensor_msgs/Image)
            /imu/data              (sensor_msgs/Imu — uses orientation yaw as the fast source)
Publishes : /heading/pool_relative (std_msgs/Float32, corrected yaw [rad], pool-axis frame)
            /heading/line_meas     (std_msgs/Float32, raw line angle [rad], debug)

Set `pool_axis_offset` so that yaw=0 points along your chosen mission axis (e.g. from
start zone toward the gate). Determine it once at the venue: hold the vehicle pointing
at the gate, read /heading/line_meas, put that value in the parameter.
"""

import math
import time
from collections import deque
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
# from sauvc_stonefish.ardupilot.modules.waf.waflib.extras.wafcache import loop
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Float32
from cv_bridge import CvBridge


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class LaneHeadingNode(Node):
    def __init__(self):
        super().__init__('lane_heading_node')
        self.declare_parameter('pool_axis_offset', 0.0)   # rad
        # pool_axis_offset (default 0.0 rad)
        # This is a calibration constant, not something computed at runtime — it's the fixed rotational offset between 
        # "the pool's grid axis" (what the lines actually measure, mod 90°) and "the direction you actually care about," 
        # e.g. straight from the start zone toward the gate.
        # Remember: the raw line-based heading is anchored to the pool's tile/lane grid — it has no idea that your 
        # mission has a meaningful "forward" direction (start → gate). If the grid happens to be rotated some 
        # arbitrary angle relative to your mission axis, the node needs to know that angle to translate "yaw 
        # relative to the grid" into "yaw relative to my mission." That's pool_axis_offset:
        # python
        # out.data = wrap(self.gyro_yaw + self.offset
        #                 - self.get_parameter('pool_axis_offset').value)
        # It's applied at the very end, as a fixed subtraction, before publishing /heading/pool_relative. The docstring 
        # tells you exactly how to determine it in practice:
        # Set pool_axis_offset so that yaw=0 points along your chosen mission axis... Determine it once at the 
        # venue: hold the vehicle pointing at the gate, read /heading/line_meas, put that value in the parameter.
        # So it's a one-time, per-venue calibration number — you physically point the vehicle at the gate, 
        # read off what the raw grid-relative measurement says at that moment, and that's the offset you 
        # plug in so that "0" in the published heading topic means "facing the gate," not "facing along 
        # some arbitrary pool tile line."
        self.declare_parameter('gain', 0.02)              # complementary gain per frame
        self.declare_parameter('min_lines', 4)
        # --- TURN HARDENING (post 360-spin + translate-and-rotate log analysis) ---
        # Validated by the controlled 360deg spin: the sign chain is CORRECT (raw lane
        # yaw tracked GT at slope -1 for a full revolution, 0 gate rejections). The
        # remaining failure mode, seen in the translate+rotate run, is STRUCTURAL to a
        # mod-90 line compass: near odd multiples of 45deg a fast, blurred turn can
        # resolve the fold to the WRONG branch; each such frame passes the 20deg gate,
        # the offset EWMA integrates the corruption, and the published yaw walks away
        # until the geometry re-agrees 90deg later. All four mitigations below are
        # gated by ONE runtime switch:
        self.declare_parameter('enable_hardening', True)
        # -p enable_hardening:=false reproduces the originally-uploaded node's fusion
        # branch EXACTLY (plain gain, latest gyro sample, no freeze, no relock) — set
        # it to compare against/roll back to that behavior without a code change.
        # When true (default), all four mitigations below are active:
        # 1) freeze_rate: while |gyro z| exceeds this, line frames MAINTAIN heading
        #    (published, counted) but never adapt the offset -> a turn can no longer
        #    drag the offset; the gyro coasts through it (its drift over a few seconds
        #    of turning is <<1deg, far below the ~20deg a wrong fold injects).
        self.declare_parameter('freeze_rate', 0.15)       # rad/s, ~8.6 deg/s
        # 2) gain scaled by line concentration R: a barely-passing R=0.6 frame (messy
        #    floor, blur) pulls at ~30% strength; a crisp R>=0.9 frame at 100%.
        self.declare_parameter('gain_r_scaling', True)
        # 3) stuck-lock recovery: if the offset DID get corrupted (e.g. hardening was
        #    off, or a slow wrong lock), the 20deg gate would otherwise reject forever.
        #    After `relock_after_rejects` consecutive gate rejections while rotating
        #    slower than `relock_max_rate`, snap the offset by the full folded error
        #    once (|err|<=45 by branch construction) and WARN. Off: relock_after_rejects<=0.
        self.declare_parameter('relock_after_rejects', 90)   # frames (~3 s at 30 fps)
        self.declare_parameter('relock_max_rate', 0.05)      # rad/s, near-stationary
        # 4) image<->yaw time alignment: `cur` is now the IMU yaw interpolated AT THE
        #    IMAGE STAMP instead of the latest sample. At 0.3 rad/s a 50 ms latency is
        #    ~0.9deg of systematic branch-selection pressure; interpolation removes it.
        #    (No parameter — strictly better when stamps are sane; falls back to the
        #    latest yaw when they are not.)
        # DEBUG WINDOW + tunable CV thresholds. Previously hardcoded (Canny 40/120,
        # Hough threshold=60, minLineLength=width//5, maxLineGap=12) with zero way to
        # see what the detector was actually looking at, or to retune without a code
        # round-trip. show_detections mirrors gate_detector_node's own param name for
        # consistency between the two vision nodes.
        self.declare_parameter('show_detections', False)
        self.declare_parameter('canny_low', 40)
        self.declare_parameter('canny_high', 120)
        self.declare_parameter('hough_threshold', 60)
        self.declare_parameter('hough_min_line_frac', 0.2)  # of image width
        self.declare_parameter('hough_max_gap', 12)
        self.bridge = CvBridge()

        self.gyro_yaw = None          # fast source: IMU yaw (drifts)
        self.offset = 0.0             # slow correction estimated from lines
        # TURN HARDENING state: (t, yaw) ring buffer for image-stamp interpolation
        # and a ~0.1 s low-passed |gyro z| so a single noisy rate sample can neither
        # trigger nor release the freeze.
        self._yaw_buf = deque(maxlen=200)   # ~4 s at 50 Hz
        self._rate_lp = 0.0
        self._reject_streak = 0
        self._n_rate_frozen = 0
        self._n_relocks = 0
        self.pub_yaw = self.create_publisher(Float32, '/heading/pool_relative', 10)
        self.pub_meas = self.create_publisher(Float32, '/heading/line_meas', 10)
        self.pub_dbg = self.create_publisher(Image, '/heading/debug_image', 2)
        self.create_subscription(Imu, '/imu/data', self.on_imu, 50)
        # FIX(QoS incompatibility): Stonefish's camera publisher uses BEST_EFFORT
        # reliability (standard for image/sensor topics). The default subscription
        # QoS is RELIABLE, and a RELIABLE subscriber cannot receive from a
        # BEST_EFFORT publisher at all — this is a hard incompatibility, not a soft
        # mismatch, and remapping the topic name alone does not fix it (confirmed:
        # "offering incompatible QoS ... Last incompatible policy: RELIABILITY").
        # qos_profile_sensor_data matches what the publisher actually uses.
        self.create_subscription(Image, '/camera_down/image_raw', self.on_image,
                                 qos_profile_sensor_data)

        # DETECTION-RATE VISIBILITY: this node previously had zero logging anywhere —
        # if the floor texture doesn't give Hough clean long edges (e.g. a mosaic tile
        # pattern rather than lane lines/grout), detect_line_angle silently returns
        # None every frame, self.offset never moves off 0, and nothing downstream can
        # tell the difference between "fusion is running and there's just nothing to
        # correct" and "fusion has never fired once." Track WHY each frame failed.
        self._n_images = 0
        self._n_too_few_lines = 0      # Hough found <min_lines segments
        self._n_low_concentration = 0  # segments found, but directions disagreed (R<0.6)
        self._n_accepted = 0           # published to /heading/line_meas
        self._n_gate_rejected = 0      # accepted line, but disagreed with current yaw >20deg
        self._last_frame_wall = None
        self._last_summary_wall = 0.0
        self.create_timer(10.0, self._heartbeat)

    def _heartbeat(self):
        now = time.time()
        if self._last_frame_wall is None:
            self.get_logger().warn(
                'lane_heading: no /camera_down/image_raw received yet — check the '
                'topic name/remap.')
            return
        if now - self._last_frame_wall > 5.0:
            self.get_logger().warn('lane_heading: image stream STALLED (>5 s).')
            return
        tot = (self._n_too_few_lines + self._n_low_concentration
               + self._n_accepted + self._n_gate_rejected)
        if tot == 0:
            return
        self.get_logger().info(
            f'lane_heading detection summary ({self._n_images} images): '
            f'accepted {self._n_accepted}, gate-rejected {self._n_gate_rejected} '
            f'(line found but disagreed with current yaw by >20 deg), '
            f'rate-frozen {self._n_rate_frozen} (turning faster than freeze_rate; '
            f'maintain-only, by design), re-locks {self._n_relocks}, '
            f'too-few-lines {self._n_too_few_lines}, low-concentration '
            f'{self._n_low_concentration} (segments found but directions disagreed, '
            f'R<0.6). current offset={math.degrees(self.offset):+.2f} deg. '
            + ('If accepted stays near 0 while images keep arriving, set '
               '-p show_detections:=true to see WHY: if the debug window shows lots '
               'of blue (raw Canny edges) but no green (surviving Hough segments), '
               'lower hough_min_line_frac and/or raise hough_max_gap — the edges are '
               'there but shorter/more broken than the current thresholds accept. If '
               'there\'s barely any blue at all, Canny itself is finding almost '
               'nothing (try lowering canny_low/canny_high, or the floor texture at '
               'this altitude may just be too fine-grained for this approach). '
               'This fusion cannot help until one of those changes, independent of '
               'anything in flow_eval_node.'
               if self._n_accepted == 0 else ''))

    def on_imu(self, msg):
        self.gyro_yaw = yaw_from_quat(msg.orientation)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if t > 0.0:
            self._yaw_buf.append((t, self.gyro_yaw))
        # low-pass |yaw rate| (alpha 0.2 at ~50 Hz -> ~0.1 s time constant)
        self._rate_lp += 0.2 * (abs(msg.angular_velocity.z) - self._rate_lp)
        # Read the associate .md file
        if self.gyro_yaw is not None:
            out = Float32()
            out.data = wrap(self.gyro_yaw + self.offset
                            - self.get_parameter('pool_axis_offset').value)
            self.pub_yaw.publish(out)

    def _yaw_at(self, t_img):
        """IMU yaw interpolated at the image timestamp (wrap-aware). Falls back to
        the latest yaw if the buffer is empty, stamps are unusable, or the image
        stamp is >0.2 s outside the buffered range (clock mismatch)."""
        # First safety check — bail out to a fallback. Two conditions:
        # t_img <= 0.0: the image's timestamp looks invalid/unset (a real 
        # ROS timestamp should be a positive number of seconds since epoch or
        # since some reference — zero or negative means something's wrong with the stamp).
        # not self._yaw_buf: the buffer is empty (e.g., no IMU messages have arrived yet).
        # In either case, there's nothing sensible to interpolate against, so just return 
        # whatever self.gyro_yaw currently is — the simple "latest known value" fallback, 
        # better than crashing or returning garbage.
        if t_img <= 0.0 or not self._yaw_buf:
            return self.gyro_yaw
        buf = self._yaw_buf
        # Second safety check — is t_img within a sane range of what we have buffered?
        # buf[0][0] is the timestamp of the oldest entry in the buffer (the deque is ordered 
        # oldest→newest since things are appended in arrival order).
        # buf[-1][0] is the timestamp of the newest entry.
        # If t_img is more than 0.2 seconds before the oldest buffered sample, or more than 0.2 
        # seconds after the newest — the image timestamp is essentially outside the range 
        # of history we have. That's a red flag (comment: "clock mismatch") — maybe the camera 
        # and IMU clocks aren't synchronized properly, or there's a large unexpected delay 
        # somewhere. Rather than trying to extrapolate wildly outside known data (unreliable), 
        # just fall back to the simple latest-yaw estimate.
        # The 0.2 second tolerance is a little slack — it allows t_img to be slightly outside 
        # the buffered range (e.g., a fraction of a millisecond due to normal jitter) without 
        # triggering the fallback, but stops far-outside timestamps from being trusted.
        if t_img <= buf[0][0] - 0.2 or t_img >= buf[-1][0] + 0.2:
            return self.gyro_yaw
        prev = None
        # Walking the buffer to find the two samples that bracket t_img. This loops through the buffer from oldest to newest.
        # prev tracks "the last sample we looked at that was still before t_img" — it starts as None because at the very 
        # first iteration, there's no "previous" sample yet.
        # The loop is looking for the first sample whose timestamp t is >= t_img — i.e., the first sample that is at 
        # or after the image's capture time. Once found, that sample (call it (t, y)) is the one right after t_img, 
        # and whatever was stored in prev just before it is the one right before t_img. Together, prev and (t, y) are 
        # the two neighboring data points that bracket the moment we care about — exactly what you need to 
        # interpolate between.
        for (t, y) in buf:
            if t >= t_img:
                # Inside the loop, once we find t >= t_img: first check — if prev is still None, 
                # that means this is the very first sample in the whole buffer, and it's already 
                # at or after t_img. There's nothing earlier to interpolate from (t_img is right 
                # at, or even before, the very start of our history) — so just return this sample's 
                # yaw y directly, no interpolation possible.
                if prev is None:
                    return y
                t0, y0 = prev
                # Otherwise, unpack prev into t0, y0 — the timestamp/yaw of the bracketing sample just before t_img. 
                # Then a defensive sanity check: if t <= t0 (the "after" sample's timestamp isn't actually later than 
                # the "before" sample's timestamp), something is inconsistent (e.g., duplicate or out-of-order timestamps) 
                # — dividing by (t - t0) in the next step would divide by zero or a negative number, so this 
                # guards against that and just returns y (the later sample) as a safe fallback rather than 
                # crashing or producing a nonsensical interpolation.
                if t <= t0:
                    return y
                f = (t_img - t0) / (t - t0) # fraction of the way from t0 to t
                return wrap(y0 + f * wrap(y - y0))
                # The actual interpolation, once we're sure it's safe to do.
                # f = (t_img - t0) / (t - t0): this is a fraction between 0 and 1 representing where t_img sits between t0 
                # and t. If t_img is exactly at t0, f = 0. If exactly at t, f = 1. If halfway between, f = 0.5.
                # wrap(y - y0): the difference in yaw between the two bracketing samples, wrap-aware. This matters because 
                # yaw is cyclic — if y0 = 170° and y = -170°, the "raw" difference (-170 - 170 = -340) looks huge, but the 
                # vehicle actually only rotated 20° (crossing the ±180° seam). wrap() correctly computes that true small 
                # difference instead of the naively-huge one.
                # y0 + f * wrap(y - y0): standard linear interpolation — start at y0, move f fraction of the way toward y 
                # (using the correctly-wrapped difference).
                # Outer wrap(...): the interpolated result itself is wrapped back into (-π, π], in case the interpolation 
                # pushed it slightly outside that canonical range.
            prev = (t, y)
        return buf[-1][1]

    def detect_line_angle(self, gray):
        """Dominant floor-line direction in IMAGE frame, in (-pi/4, pi/4]. None if unsure.

        Sets self._last_edges/self._last_lines/self._last_R for the debug window,
        and self._last_reject ('too_few_lines' or 'low_concentration') on failure,
        so on_image and the debug overlay can both see WHY, not just that it failed."""
        canny_low = self.get_parameter('canny_low').value
        canny_high = self.get_parameter('canny_high').value
        hough_thresh = self.get_parameter('hough_threshold').value
        min_line_len = int(gray.shape[1] * self.get_parameter('hough_min_line_frac').value)
        max_gap = self.get_parameter('hough_max_gap').value

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, canny_low, canny_high)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=hough_thresh,
                                minLineLength=min_line_len, maxLineGap=max_gap)
        self._last_edges = edges
        self._last_lines = lines
        self._last_R = None
        if lines is None or len(lines) < self.get_parameter('min_lines').value:
            self._last_reject = 'too_few_lines'
            return None
        # Fold every segment angle into mod-90° space and take a length-weighted
        # circular mean (period pi/2 -> multiply angles by 4).
        s = c = 0.0
        for (x1, y1, x2, y2) in lines[:, 0]:
            ang = math.atan2(y2 - y1, x2 - x1)          # (-pi, pi]
            L = math.hypot(x2 - x1, y2 - y1)
            s += L * math.sin(4 * ang)
            c += L * math.cos(4 * ang)
        if s == 0 and c == 0:
            self._last_reject = 'too_few_lines'
            return None
        mean4 = math.atan2(s, c)
        # concentration check: reject frames where line directions disagree wildly
        R = math.hypot(s, c) / sum(math.hypot(x2 - x1, y2 - y1)
                                   for (x1, y1, x2, y2) in lines[:, 0])
        self._last_R = R
        if R < 0.6:
            self._last_reject = 'low_concentration'
            return None
        return mean4 / 4.0                               # (-pi/4, pi/4]

    def _draw_debug(self, gray, status):
        """Blue = every Canny edge pixel (shows what's there before length filtering).
        Green = the Hough segments that survived minLineLength/maxLineGap (what the
        angle/R computation actually used). Seeing lots of blue with no green means
        the edges exist but are shorter/more broken than the thresholds accept;
        almost no blue at all means Canny itself isn't finding much — two different
        problems with two different fixes, which is exactly what was impossible to
        tell apart from the terminal summary alone."""
        dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        edges = getattr(self, '_last_edges', None)
        if edges is not None:
            dbg[edges > 0] = (255, 80, 0)
        lines = getattr(self, '_last_lines', None)
        n = 0 if lines is None else len(lines)
        if lines is not None:
            for (x1, y1, x2, y2) in lines[:, 0]:
                cv2.line(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)
        R = getattr(self, '_last_R', None)
        min_lines = self.get_parameter('min_lines').value
        txt1 = f"segments: {n}/{min_lines}  R: {'--' if R is None else f'{R:.2f}'}/0.60"
        txt2 = f"status: {status}  offset: {math.degrees(self.offset):+.1f} deg"
        txt3 = (f"rate: {math.degrees(self._rate_lp):+.1f} deg/s  "
                f"(freeze > {math.degrees(float(self.get_parameter('freeze_rate').value)):.0f})")
        cv2.putText(dbg, txt1, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(dbg, txt2, (8, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv2.putText(dbg, txt3, (8, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return dbg

    def on_image(self, msg):
        self._last_frame_wall = time.time()
        self._n_images += 1
        if self.gyro_yaw is None:
            return
        gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        self._last_reject = None
        # why setting the following variables to None? 
        # Because they are used in the debug drawing function to show the last edges, 
        # lines, and concentration R. If we don't reset them here, they might show 
        # stale data from a previous frame, which could be misleading. By setting 
        # them to None at the start of processing a new image, we ensure that if 
        # the current frame fails to detect any lines or edges, the debug output 
        # will accurately reflect that no valid data was found for this frame.
        self._last_edges = self._last_lines = self._last_R = None
        ang = self.detect_line_angle(gray)
        status = 'accepted'
        if ang is None:
            if self._last_reject == 'low_concentration':
                self._n_low_concentration += 1
                status = 'low-concentration'
            else:
                self._n_too_few_lines += 1
                status = 'too-few-lines'
        else:
            self.pub_meas.publish(Float32(data=float(ang)))
            hardening = bool(self.get_parameter('enable_hardening').value)
            # Lines at image angle `ang` mean vehicle yaw relative to the pool grid is
            # -ang (mod 90°) — sign VALIDATED by the 360° spin test (slope -1 over a
            # full revolution). Pick the mod-90 branch closest to the current yaw.
            if hardening:
                # Yaw AT THE IMAGE STAMP: using the latest IMU sample instead biases
                # branch selection by rate*latency, exactly when the 45° fold is closest.
                yaw_img = self._yaw_at(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)
                if yaw_img is None:
                    return
            else:
                # ORIGINAL behavior: latest gyro sample, no stamp interpolation.
                yaw_img = self.gyro_yaw
            cur = wrap(yaw_img + self.offset)
            meas = -ang
            k = round((cur - meas) / (math.pi / 2))
            meas_unwrapped = meas + k * (math.pi / 2)
            err = wrap(meas_unwrapped - cur)
            turning = hardening and self._rate_lp > float(self.get_parameter('freeze_rate').value)
            if turning:
                # MAINTAIN, never adapt, while turning: a fast turn is precisely when
                # the mod-90 fold can resolve to the wrong branch (blur + lag), and one
                # wrong lock used to drag the offset for the rest of the leg. The gyro
                # coasts through the turn instead; adaptation resumes when the rate
                # drops. Not counted as accepted OR rejected — it is neither.
                self._n_rate_frozen += 1
                status = (f'frozen (turning {math.degrees(self._rate_lp):.0f} deg/s '
                          f'> freeze_rate)')
            elif abs(err) < math.radians(20):             # sanity gate
                g = float(self.get_parameter('gain').value)
                if hardening and bool(self.get_parameter('gain_r_scaling').value):
                    # R in [0.6, 1) here (0.6 floor enforced in detect_line_angle).
                    # Barely-coherent frames pull at ~30%, crisp ones at 100%.
                    R = self._last_R if self._last_R is not None else 0.6
                    g *= min(1.0, max(0.3, (R - 0.6) / 0.3))
                self.offset = wrap(self.offset + g * err)
                self._n_accepted += 1
                self._reject_streak = 0
                status = 'ACCEPTED'
            else:
                self._n_gate_rejected += 1
                self._reject_streak += 1
                status = f'gate-rejected ({math.degrees(err):+.0f} deg)'
                # STUCK-LOCK RECOVERY (hardening only): a corrupted offset makes every
                # future frame gate-reject (the fold keeps |err| <= 45, always > 20).
                # Only re-lock when nearly stationary (a wrong fold is then implausible:
                # lines are crisp and branch selection unambiguous) and persistent.
                n_need = int(self.get_parameter('relock_after_rejects').value)
                if (hardening and n_need > 0 and self._reject_streak >= n_need
                        and self._rate_lp < float(self.get_parameter('relock_max_rate').value)):
                    self.offset = wrap(self.offset + err)
                    self._n_relocks += 1
                    self._reject_streak = 0
                    self.get_logger().warn(
                        f'lane_heading: offset RE-LOCKED by {math.degrees(err):+.1f} deg '
                        f'after {n_need} consecutive gate rejections while near-'
                        f'stationary (mod-90 branch: a residual multiple of 90 deg '
                        f'cannot be detected here — verify heading before trusting).')
                    status = 'RE-LOCKED'

        if self.get_parameter('show_detections').value or self.pub_dbg.get_subscription_count() > 0:
            dbg = self._draw_debug(gray, status)
            self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8'))
            if self.get_parameter('show_detections').value:
                cv2.imshow('lane_heading', dbg)
                cv2.waitKey(1)


def main():
    rclpy.init()
    node = LaneHeadingNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
