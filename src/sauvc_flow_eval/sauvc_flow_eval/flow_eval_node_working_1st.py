#!/usr/bin/env python3
"""flow_eval_node — compares localization approaches against Stonefish ground truth.

Runs FIVE estimators off the same live sensor stream and publishes each as its own
nav_msgs/Odometry on /eval/*, all in ONE common frame (default NED) so PlotJuggler can
overlay them directly:

  /eval/ground_truth   Stonefish odometry (the reference)
  /eval/flow           flow + altitude (metric via pressure), integrated to position
  /eval/ekf            self-contained EKF fusing flow + depth
  /eval/pressure       depth only (world Z; x,y left at 0)
  /eval/gtsam          IMU preintegration, integrated to position   [only if gtsam present]

Also opens two OpenCV windows: the raw down-camera feed and the optical-flow overlay.

DESIGN (per request):
  * Landmark/gate localization is deliberately NOT included — this is dead-reckoning-style
    comparison only.
  * Each estimator lives in its own module under estimators/; this node only wires them.
  * No ArduSub/SITL anywhere.
  * Modular for the real vehicle: it subscribes to the SHIMMED topics (/imu/data, /depth,
    /altitude, /camera_down/image_raw) which are identical on the real AUV, plus the
    sim-only /sauvc_auv/odometry for ground truth (absent on hardware -> that estimator
    simply stays silent).

FRAMES: `compare_frame` (default 'ned'). Ground truth is native NED; the ENU-native
estimates are converted with the tested sauvc_sim_bridge.frames conversion, in ONE place
(eval_common), never ad-hoc. See eval_common for the full rationale.

FIXED (drift post-mortem, see log analysis):
  1. FLU->FRD body flip before rotating flow velocity by a NED yaw (was mirroring the
     lateral/east legs for flow, EKF and GTSAM simultaneously).
  2. iSAM2 keyframes throttled to gtsam_keyframe_period (was a full isam.update() per
     30 Hz camera frame in the single callback thread -> frame drops -> flow dropouts).
  3. Flow dropouts (flow_core returning None) are now counted and WARNed.
  4. Gyro derotation uses the hardware-validated (-wy, -wx) camera mapping.
  5. Default intrinsics now match the 640x480 scene (resolution parity fix).

UPGRADES (integrated in this version):
  A. SELF-SUFFICIENT ALTITUDE (self_altitude:=true, default). The floor-profile
     altitude has now failed THREE runs in a row because depth_shim's odometry feed
     was never connected at launch. This node already subscribes to ground-truth
     odometry and pressure, so it now computes the camera-to-floor altitude ITSELF:
     V-floor profile at the GT x, minus (sensor depth + mount offsets), with tilt
     compensation. /altitude is only used as a cross-check; a persistent mismatch
     is WARNed as a shim misconfiguration diagnosis.
  B. Tilt compensation: range along the optical axis = altitude / (cos r * cos p).
  C. Timestamped gyro sync: IMU ring buffer, derotation rates interpolated at the
     inter-frame midpoint instead of using the latest sample.
  D. Optional lane-heading yaw fusion (use_lane_heading) from /heading/pool_relative.
  E. Quality-scaled EKF measurement noise (flow_velocity_node's variance model).
  F. ZUPT: flow + gyro both ~0 -> velocity clamped to exactly 0 (kills stationary creep).
  G. CLAHE + grid-distributed features in flow_core (use_clahe, feature_grid_*).
"""

import math
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, Image, FluidPressure
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry

try:
    from cv_bridge import CvBridge
    import cv2
    _HAVE_CV = True
except Exception:
    _HAVE_CV = False

from sauvc_flow_eval.eval_common import (
    PositionIntegrator, depth_to_world_z, to_compare_frame_world, gt_world_to_compare)
from sauvc_flow_eval.estimators.flow_estimator import FlowEstimator
from sauvc_flow_eval.estimators.ekf_estimator import EkfEstimator
from sauvc_flow_eval.estimators.gtsam_estimator import GtsamEstimator

from sauvc_sim_bridge.frames import ned_frd_quat_to_enu_flu, flu_frd_to_ned_wxyz


def _enu_quat_to_ned_wxyz(x, y, z, w):
    return flu_frd_to_ned_wxyz(x, y, z, w)


def _parse_scene(path):
    """Parse a Stonefish .scn: vehicle start pose + landmark world positions.

    Returns (start_xyz_ned | None, start_yaw, {name: (x, y, z)}). Landmarks are every
    <static> or <dynamic> whose name matches gate/flare/drum/tub, plus a derived
    'GateCenter' (midpoint of GatePostPort/GatePostStbd - the map point whose x is the
    rulebook-known gate line). $(find ...) substitutions only appear in file paths, so
    the attributes needed here parse as literal XML.
    """
    import re
    import xml.etree.ElementTree as ET
    root = ET.fromstring(open(path).read())
    start_xyz, start_yaw = None, 0.0
    for inc in root.iter('include'):
        f = inc.get('file', '')
        if 'my_auv' in f or 'vehicle' in f:
            for arg in inc.iter('arg'):
                if arg.get('name') == 'start_position':
                    start_xyz = tuple(float(v) for v in arg.get('value').split())
                elif arg.get('name') == 'start_yaw':
                    start_yaw = float(arg.get('value'))
    pat = re.compile(r'gate|flare|drum|tub', re.I)
    landmarks = {}
    for tag in ('static', 'dynamic'):
        for el in root.iter(tag):
            name = el.get('name', '')
            wt = el.find('world_transform')
            if name and pat.search(name) and wt is not None:
                landmarks[name] = tuple(float(v) for v in wt.get('xyz').split())
    if 'GatePostPort' in landmarks and 'GatePostStbd' in landmarks:
        p_, s_ = landmarks['GatePostPort'], landmarks['GatePostStbd']
        landmarks['GateCenter'] = tuple((a + b) / 2 for a, b in zip(p_, s_))
    return start_xyz, start_yaw, landmarks


def _yaw_from_quat_xyzw(x, y, z, w):
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class FlowEvalNode(Node):
    def __init__(self):
        super().__init__('flow_eval_node')
        p = self.declare_parameter
        p('compare_frame', 'ned')          # 'ned' (default, per request) | 'enu'
        p('robot_name', 'sauvc_auv')
        # FIX(resolution parity): scene camera is now 640x480 (see my_auv.scn + the
        # README "Resolution parity" note). Stonefish intrinsics are analytic:
        #   fx = fy = (640/2)/tan(40deg) = 381.36, cx = 320, cy = 240.
        # If you run the old 1280x720 scene, override these back to 762.72/640/360.
        p('fx', 381.36); p('fy', 381.36); p('cx', 320.0); p('cy', 240.0)
        p('pool_depth', 1.6)               # for altitude if /altitude is unavailable
        # FIX(dropout/CPU): iSAM2 keyframe period. add_keyframe used to run a FULL
        # isam.update() on EVERY 30 Hz camera frame in this same callback thread,
        # starving the sensor-data QoS queue -> dropped frames -> LK failure -> the
        # 48% flow freezes measured in the log. IMU preintegration still accumulates
        # at IMU rate between keyframes, so nothing is lost by throttling.
        p('gtsam_keyframe_period', 0.2)    # seconds (~5 Hz keyframes)
        # --- SCENE PARSER: start pose + landmark map straight from the .scn ---
        # If set, the node parses the Stonefish scene XML for the vehicle's
        # start_position/start_yaw and every Gate/Flare/Drum/Tub landmark pose.
        # Uses: (a) anchor fallback BEFORE the first ground-truth message (and on
        # hardware, where ground truth never exists) so a changed spawn needs no code
        # edit; (b) self.landmarks = the known map for landmark EKF updates and for
        # landmark_truth_node cross-checks. NOTE: with ground truth present, all four
        # tracks are ALREADY anchored to GT's first pose automatically - a changed
        # scene start needs nothing at all in that case.
        p('scene_file', '')
        # --- LANDMARK LOCALIZATION (consumes /vision/features from gate_detector) ---
        # 'off'  : dead reckoning only (previous behavior).
        # 'gate' : rulebook-known gate x (from the parsed scene / gate_x fallback) ->
        #          anisotropic x-only correction of EKF + GTSAM on every gated
        #          gate observation. No mapping. The biggest win, do this first.
        # 'map'  : 'gate' PLUS the small semantic map: every uniquely-named feature's
        #          world position is frozen after lm_min_frames quality-gated sightings
        #          within lm_max_first_range; later re-observations correct BOTH axes
        #          with var = stored_var + observation_var (decoupled map, deliberately
        #          NOT SLAM state augmentation; stored var is inflated to compensate).
        p('landmark_mode', 'off')
        p('features_topic', '/vision/features')
        p('gate_x', 4.4)                  # fallback if scene_file wasn't given
        p('lm_min_frames', 5)             # sightings before a landmark is frozen
        p('lm_max_first_range', 8.0)      # m, only near sightings shape the first fix
        p('lm_innov_gate', 2.0)           # m, reject corrections larger than this
        p('lm_obs_sigma', 0.05)           # observation sigma = lm_obs_sigma * range
        p('lm_map_inflate', 1.5)          # stored-variance inflation (decoupling tax)
        # --- UPGRADE A: self-sufficient altitude (see docstring) ---
        p('self_altitude', True)           # compute camera altitude in-node
        p('use_floor_profile', True)       # V-floor; false -> flat pool_depth
        p('floor_profile_x', [0.0, 12.5, 25.0])     # wall-referenced breakpoints [m]
        p('floor_profile_depth', [1.2, 1.6, 1.2])   # floor depth at breakpoints [m]
        p('profile_x_offset', 12.5)        # world NED x -> wall-referenced profile x
        p('sensor_above_origin', 0.10)     # pressure sensor mount (my_auv.scn)
        p('camera_below_origin', 0.11)     # down camera mount (my_auv.scn)
        p('alt_mismatch_warn', 0.15)       # warn if /altitude differs by more [m]
        # --- UPGRADE B: tilt compensation ---
        p('tilt_compensation', True)
        # --- UPGRADE D: lane-heading yaw fusion (off unless lane_heading_node runs) ---
        p('use_lane_heading', False)
        p('lane_heading_topic', '/heading/pool_relative')
        p('pool_axis_ned_yaw', 0.0)        # NED yaw of the pool axis lane yaw=0 [rad]
        p('lane_fresh_s', 1.0)             # max staleness to trust a lane yaw
        # --- UPGRADE E: quality-scaled EKF noise ---
        p('r_flow_base', 0.02)             # base variance, scaled by frame quality
        # --- FLOW->IMU YAW ALIGNMENT (fixes the y-drift-while-moving-in-x) ---
        # flow_core is cross-coupling-free (verified) and PositionIntegrator is a clean
        # rotation, so a lateral leak that grows with FORWARD distance (not time) can
        # only be a fixed yaw misalignment between the flow body frame and the yaw used
        # to rotate it into the world — i.e. an IMU-shim NED-yaw bias or the down-
        # camera/IMU mount yaw in the .scn. This is the extrinsic every DVL/flow system
        # is calibrated for. flow_yaw_offset [rad] is added to the yaw used for flow in
        # ALL THREE flow-based estimators (EKF, integrator, GTSAM). On hardware set it
        # from a calibration run and disable autocal. In sim, autocal measures it from
        # ground truth on straight legs, applies it, and prints the number + attribution.
        p('flow_yaw_offset', 0.0)          # rad, fixed flow-body -> IMU-yaw correction
        p('flow_yaw_autocal', True)        # sim only: estimate offset from GT straight legs
        p('flow_yaw_cal_min_speed', 0.05)  # m/s, only sample when clearly translating
        p('flow_yaw_cal_freeze_n', 60)     # freeze the estimate after this many samples
        # --- UPGRADE F: ZUPT ---
        p('zupt', True)
        p('zupt_vel', 0.03)                # m/s: below this AND
        p('zupt_gyro', 0.02)               # rad/s: below this -> clamp v to 0
        # --- UPGRADE G: texture robustness (implemented in flow_core) ---
        # use_clahe default OFF: CLAHE's space-variant equalization breaks held-gap
        # recovery (measured); enable only for genuinely washed-out footage.
        p('use_clahe', False)
        p('feature_grid_rows', 3)
        p('feature_grid_cols', 4)
        p('show_windows', True)          # master: any OpenCV window at all
        p('show_optical_flow', True)     # the optical-flow overlay window specifically
        p('show_camera', True)           # the raw down-camera window specifically
        p('print_estimates', True)       # print all 5 x/y/z to the terminal
        p('print_rate', 5.0)             # Hz, terminal print throttle
        p('gravity', 9.81)

        g = lambda n: self.get_parameter(n).value
        self.frame = g('compare_frame')
        if self.frame not in ('ned', 'enu'):
            raise ValueError("compare_frame must be 'ned' or 'enu'")
        self.show = g('show_windows') and _HAVE_CV
        self.show_flow = self.show and g('show_optical_flow')
        self.show_cam = self.show and g('show_camera')
        self.print_est = g('print_estimates')
        self.print_period = 1.0 / max(g('print_rate'), 0.1)
        self.kf_period = float(g('gtsam_keyframe_period'))
        self._last_kf_t = 0.0
        # FIX(dropout visibility): count flow failures and WARN, so a 48% dropout
        # rate can never again hide inside a clean-looking log.
        self._drop_count = 0
        self._drop_streak = 0
        self._last_drop_warn = 0.0
        self.self_alt = bool(g('self_altitude'))
        self.use_profile = bool(g('use_floor_profile'))
        self.prof_x = np.asarray(g('floor_profile_x'), float)
        self.prof_d = np.asarray(g('floor_profile_depth'), float)
        self.prof_off = float(g('profile_x_offset'))
        self.sensor_above = float(g('sensor_above_origin'))
        self.cam_below = float(g('camera_below_origin'))
        self.alt_warn_m = float(g('alt_mismatch_warn'))
        self.tilt_comp = bool(g('tilt_compensation'))
        self.use_lane = bool(g('use_lane_heading'))
        self.pool_axis_ned = float(g('pool_axis_ned_yaw'))
        self.lane_fresh_s = float(g('lane_fresh_s'))
        self.r_flow_base = float(g('r_flow_base'))
        # flow->IMU yaw alignment state
        self.flow_yaw_offset = float(g('flow_yaw_offset'))
        self.flow_yaw_autocal = bool(g('flow_yaw_autocal'))
        self.yawcal_min_speed = float(g('flow_yaw_cal_min_speed'))
        self.yawcal_freeze_n = int(g('flow_yaw_cal_freeze_n'))
        self._yawcal_s = 0.0            # sum sin(offset_sample)  (circular mean)
        self._yawcal_c = 0.0            # sum cos(offset_sample)
        self._yawcal_n = 0
        self._yawcal_frozen = not self.flow_yaw_autocal
        self._gt_vel = None            # ground-truth planar velocity (compare frame)
        self._gt_prev_xy = None
        self._gt_prev_t = None
        self.zupt_on = bool(g('zupt'))
        self.zupt_vel = float(g('zupt_vel'))
        self.zupt_gyro = float(g('zupt_gyro'))
        self._alt_warned_t = 0.0
        self._zupt_count = 0
        # UPGRADE C: IMU ring buffer for midpoint gyro interpolation (~4 s @ 100 Hz)
        self._gyro_buf = deque(maxlen=400)      # (t, wx, wy, wz)
        self.roll = 0.0
        self.pitch = 0.0
        self.gt_x_ned = None                    # raw NED x for the floor profile
        # SCENE PARSER results
        self.scene_start_ned = None             # (x, y, z) from the .scn, NED
        self.scene_start_yaw = 0.0
        self.landmarks = {}                     # name -> (x, y, z) NED world
        self.lm_mode = g('landmark_mode')
        self.gate_x_known = float(g('gate_x'))
        self.lm_min_frames = int(g('lm_min_frames'))
        self.lm_max_first_range = float(g('lm_max_first_range'))
        self.lm_innov_gate = float(g('lm_innov_gate'))
        self.lm_obs_sigma = float(g('lm_obs_sigma'))
        self.lm_map_inflate = float(g('lm_map_inflate'))
        self.lm_map = {}          # name -> dict(sum, n, pos, var, frozen)
        self._lm_log_t = 0.0
        self._lane_warned = 0.0
        sf = g('scene_file')
        if sf:
            try:
                self.scene_start_ned, self.scene_start_yaw, self.landmarks = \
                    _parse_scene(sf)
                self.get_logger().info(
                    f'scene parsed: start={self.scene_start_ned} '
                    f'yaw={self.scene_start_yaw} | '
                    f'{len(self.landmarks)} landmarks: '
                    + ', '.join(sorted(self.landmarks)))
            except Exception as e:
                self.get_logger().error(f'scene_file parse failed ({e}) - '
                                        'continuing without scene info')
        self.lane_yaw = None                    # latest lane-heading yaw [rad]
        self.lane_yaw_t = -1e9
        # latest published value per source, for the terminal table
        self._latest = {k: None for k in
                        ('ground_truth', 'flow', 'ekf', 'pressure', 'gtsam')}
        self._last_print = 0.0

        # --- estimators ---
        self.flow = FlowEstimator(g('fx'), g('fy'), g('cx'), g('cy'),
                                  use_clahe=g('use_clahe'),
                                  grid_rows=g('feature_grid_rows'),
                                  grid_cols=g('feature_grid_cols'))
        self.ekf = EkfEstimator()
        self.gtsam = GtsamEstimator(gravity=g('gravity'), compare_frame=self.frame)
        self.flow_pos = PositionIntegrator()
        self.gtsam_pos = PositionIntegrator()

        if self.gtsam.available:
            self.get_logger().info('gtsam present -> /eval/gtsam active')
        else:
            self.get_logger().warn('gtsam NOT available -> /eval/gtsam disabled '
                                   '(other four estimators run normally)')

        # --- state caches ---
        self.bridge = CvBridge() if _HAVE_CV else None
        self.gyro_body = (0.0, 0.0, 0.0)
        self.accel_body = (0.0, 0.0, 0.0)
        self.last_quat_wxyz_ned = (1.0, 0.0, 0.0, 0.0)
        self._have_imu = False   # FIX: never seed gtsam attitude from the identity
                                 # placeholder — with a wrong attitude the 9.81 m/s^2
                                 # specific force integrates as motion.
        self.yaw = 0.0          # compare-frame yaw (for flow integration / EKF)
        self.yaw_ned = 0.0      # NED yaw (for rotating flow into the graph's NED world)
        self.gt_anchor = None   # ground-truth first pose (compare frame); all tracks start here
        self.gt_yaw_ned = None  # ground-truth NED yaw (orientation), for the verification print below
        self.depth = 0.0
        self.altitude = None
        self.prev_img_t = None

        # --- publishers ---
        self.pubs = {k: self.create_publisher(Odometry, f'/eval/{k}', 10)
                     for k in ('ground_truth', 'flow', 'ekf', 'pressure', 'gtsam')}

        # --- subscriptions ---
        robot = g('robot_name')
        self.create_subscription(Imu, '/imu/data', self.on_imu, qos_profile_sensor_data)
        self.create_subscription(FluidPressure, f'/{robot}/pressure',
                                 self.on_pressure, qos_profile_sensor_data)
        self.create_subscription(Float32, '/altitude', self.on_altitude, 10)
        self.create_subscription(Image, '/camera_down/image_raw',
                                 self.on_image, qos_profile_sensor_data)
        self.create_subscription(Odometry, f'/{robot}/odometry',
                                 self.on_ground_truth, qos_profile_sensor_data)
        if self.use_lane:
            self.create_subscription(Float32, g('lane_heading_topic'),
                                     self.on_lane_heading, 10)
        if self.lm_mode in ('gate', 'map'):
            from std_msgs.msg import String as _String
            self.create_subscription(_String, g('features_topic'),
                                     self.on_feature, 20)
            if 'GateCenter' in self.landmarks:
                self.gate_x_known = float(self.landmarks['GateCenter'][0])
            self.get_logger().info(
                f"landmark_mode='{self.lm_mode}': gate x = {self.gate_x_known:.2f} "
                f"(from {'scene' if 'GateCenter' in self.landmarks else 'gate_x param'})"
                + (', small-map ON' if self.lm_mode == 'map' else ''))

        if self.show_cam:
            cv2.namedWindow('down camera', cv2.WINDOW_NORMAL)
        if self.show_flow:
            cv2.namedWindow('optical flow', cv2.WINDOW_NORMAL)
        self.get_logger().info(
            f"flow_eval up, compare_frame='{self.frame}'. Publishing /eval/* "
            "(ground_truth, flow, ekf, pressure"
            + (", gtsam" if self.gtsam.available else "") + "). "
            + ("ALTITUDE: self-computed in-node (floor profile at GT x, camera datum, "
               "tilt-compensated); /altitude used only as a cross-check."
               if self.self_alt else "ALTITUDE: from /altitude (shim)."))

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
        alt = max(self._floor_depth_at(self.gt_x_ned) - cam_depth, 0.05)
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

    def _update_yaw_autocal(self, vx_b, vy_b, yaw_raw):
        """Estimate the fixed flow-body -> IMU-yaw offset from ground truth on straight
        legs. Rotates the RAW flow body velocity (no offset) into the compare-frame
        world with yaw_raw, and measures the angle to the GT world-velocity direction.
        Averages circularly and freezes; sim-only (needs GT), harmless on hardware."""
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
        self._yawcal_s += math.sin(d)
        self._yawcal_c += math.cos(d)
        self._yawcal_n += 1
        if self._yawcal_n >= 5:              # apply the running mean once it's stable-ish
            self.flow_yaw_offset = math.atan2(self._yawcal_s, self._yawcal_c)
        if self._yawcal_n >= self.yawcal_freeze_n:
            self._yawcal_frozen = True
            gt_hdg = (math.degrees(math.atan2(gv[1], gv[0]))
                      if self.frame == 'ned' else float('nan'))
            self.get_logger().warn(
                'flow->IMU yaw offset auto-calibrated to %+.2f deg (%.4f rad) over %d '
                'straight-line samples. This is a FIXED frame misalignment between the '
                'flow body frame and the yaw rotating it — an IMU-shim NED-yaw bias or '
                'the down-camera/IMU mount yaw in the .scn — NOT vehicle motion, which '
                'is why every flow-based track drifted in y while moving only in x. '
                'It is now nulled for all three estimators. To fix it at the source (and '
                'remove the startup transient): set -p flow_yaw_offset:=%.4f '
                '-p flow_yaw_autocal:=false, or correct the mount/shim. Attribution: '
                'IMU yaw_ned=%.2f deg vs GT heading=%.2f deg — if these disagree it is '
                'the IMU/shim; if they agree it is the camera mount.'
                % (math.degrees(self.flow_yaw_offset), self.flow_yaw_offset,
                   self._yawcal_n, self.flow_yaw_offset,
                   math.degrees(self.yaw_ned), gt_hdg))

    def _fused_yaws(self, t_wall):
        """UPGRADE D: (yaw_compare, yaw_ned) with fresh lane-heading substitution."""
        yaw_ned = self.yaw_ned
        if (self.use_lane and self.lane_yaw is not None
                and t_wall - self.lane_yaw_t < self.lane_fresh_s):
            cand = self.lane_yaw + self.pool_axis_ned
            # SANITY GATE(frame mismatch): lane_heading's yaw is built from the
            # ENU/FLU IMU quaternion; if pool_axis_offset was not set to the pool
            # axis's ENU yaw (pi/2 for this scene), the published yaw is off by a
            # multiple of 90 deg and would rotate x motion into y. Reject and WARN
            # instead of corrupting the track.
            d = (cand - self.yaw_ned + np.pi) % (2 * np.pi) - np.pi
            if abs(d) > np.radians(30.0):
                if t_wall - self._lane_warned > 5.0:
                    self._lane_warned = t_wall
                    self.get_logger().warn(
                        f'lane yaw rejected: disagrees with IMU-NED yaw by '
                        f'{np.degrees(d):+.0f} deg — run lane_heading_node with '
                        '-p pool_axis_offset:=1.5708 (the pool axis ENU yaw). '
                        'Falling back to IMU yaw.')
            else:
                yaw_ned = cand
        yaw_cmp = yaw_ned if self.frame == 'ned' else (np.pi / 2 - yaw_ned)
        return yaw_cmp, yaw_ned

    # ---- landmark localization (consumes gate_detector's /vision/features) ----
    def on_feature(self, msg):
        try:
            name, bx, by, bz, rng, brg, elev, area = msg.data.split(',')
            bx, by, rng = float(bx), float(by), float(rng)
        except ValueError:
            return
        t = self.get_clock().now().nanoseconds * 1e-9
        # body FRD -> NED world (planar; pitch/roll are small at cruise)
        c, s = np.cos(self.yaw_ned), np.sin(self.yaw_ned)
        rwx = c * bx - s * by
        rwy = s * bx + c * by
        obs_var = (self.lm_obs_sigma * max(rng, 1.0)) ** 2

        ex, ey = self.ekf.x[0], self.ekf.x[1]     # current EKF position estimate

        # ---- GATE: rulebook-known x -> anisotropic absolute correction ----
        if name.startswith('GatePost') or name == 'GateCenter':
            p_meas_x = self.gate_x_known - rwx
            if abs(p_meas_x - ex) < self.lm_innov_gate:
                self.ekf.update_position_xy(p_meas_x, ey, t, obs_var, 1e12)
                self.gtsam.add_landmark_xy(p_meas_x, ey, np.sqrt(obs_var), 1e6)
                if t - self._lm_log_t > 2.0:
                    self._lm_log_t = t
                    self.get_logger().info(
                        f'GATE x-correction applied: x <- {p_meas_x:+.2f} '
                        f'(innov {p_meas_x - ex:+.2f} m, sigma '
                        f'{np.sqrt(obs_var):.2f})')
            if self.lm_mode != 'map':
                return
            name = 'GateCenter'                    # map the pair as one landmark

        if self.lm_mode != 'map':
            return

        # ---- SMALL MAP: freeze at first quality-gated sightings, then correct ----
        lm = self.lm_map.setdefault(
            name, dict(sum=np.zeros(2), n=0, pos=None, var=None, frozen=False))
        if not lm['frozen']:
            if rng <= self.lm_max_first_range:
                lm['sum'] += np.array([ex + rwx, ey + rwy])
                lm['n'] += 1
                if lm['n'] >= self.lm_min_frames:
                    lm['pos'] = lm['sum'] / lm['n']
                    lm['var'] = obs_var * self.lm_map_inflate
                    lm['frozen'] = True
                    self.get_logger().info(
                        f'MAP: {name} frozen at ({lm["pos"][0]:+.2f}, '
                        f'{lm["pos"][1]:+.2f}) after {lm["n"]} sightings '
                        f'(sigma {np.sqrt(lm["var"]):.2f} m) — absolute error floor '
                        'is the vehicle pose error at THESE sightings.')
            return
        # re-observation of a frozen landmark -> both-axis correction
        p_meas = lm['pos'] - np.array([rwx, rwy])
        innov = np.hypot(p_meas[0] - ex, p_meas[1] - ey)
        if innov < self.lm_innov_gate:
            v = lm['var'] + obs_var
            self.ekf.update_position_xy(p_meas[0], p_meas[1], t, v, v)
            self.gtsam.add_landmark_xy(p_meas[0], p_meas[1], np.sqrt(v), np.sqrt(v))
            if t - self._lm_log_t > 2.0:
                self._lm_log_t = t
                self.get_logger().info(
                    f'MAP re-observation: {name} -> pos <- ({p_meas[0]:+.2f}, '
                    f'{p_meas[1]:+.2f}), innov {innov:.2f} m')

    # ---- sensor callbacks ----
    def on_imu(self, msg):
        # /imu/data is already ENU/FLU (imu_shim). For the GTSAM path we want body-frame
        # accel/gyro, which are frame-of-the-body regardless of world convention, so use
        # them directly. yaw is taken in the compare frame.
        q = msg.orientation
        self._have_imu = True
        self.gyro_body = (msg.angular_velocity.x, msg.angular_velocity.y,
                          msg.angular_velocity.z)
        self.accel_body = (msg.linear_acceleration.x, msg.linear_acceleration.y,
                           msg.linear_acceleration.z)
        # cache the body->NED quaternion (w,x,y,z) for gtsam gravity-aligned init.
        # /imu/data is ENU/FLU; convert its quat to NED/FRD via the tested frames path.
        self.last_quat_wxyz_ned = _enu_quat_to_ned_wxyz(q.x, q.y, q.z, q.w)
        # UPGRADE C: ring-buffer the gyro with its stamp for midpoint interpolation.
        ti = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self._gyro_buf.append((ti,) + self.gyro_body)
        # UPGRADE B: roll/pitch from the ENU/FLU quaternion for tilt compensation.
        self.roll = math.atan2(2.0 * (q.w * q.x + q.y * q.z),
                               1.0 - 2.0 * (q.x * q.x + q.y * q.y))
        sp = 2.0 * (q.w * q.y - q.z * q.x)
        self.pitch = math.asin(max(-1.0, min(1.0, sp)))
        yaw_enu = _yaw_from_quat_xyzw(q.x, q.y, q.z, q.w)
        # yaw in NED = 90 - yaw_enu wrapping; simplest correct route is to rotate a
        # heading vector. For planar integration only the world we integrate in matters:
        self.yaw_ned = np.pi / 2 - yaw_enu     # ENU yaw (CCW/East) -> NED yaw (CW/North)
        self.yaw = self.yaw_ned if self.frame == 'ned' else yaw_enu
        # feed GTSAM preintegration at IMU rate
        if self.gtsam.available:
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.gtsam.add_imu(self.accel_body, self.gyro_body, t)

    def on_pressure(self, msg):
        # depth from gauge pressure (rho*g matches the scene; see depth_shim)
        self.depth = msg.fluid_pressure / (1000.0 * 9.81)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pz = depth_to_world_z(self.depth, self.frame)
        self.ekf.update_depth(pz, t)
        # /eval/pressure: depth only, x=y=0
        ax, ay, az = self._anchor(0.0, 0.0, pz)
        self._publish('pressure', ax, ay, az, msg.header.stamp)

    def on_altitude(self, msg):
        self.altitude = msg.data

    def on_lane_heading(self, msg):
        # UPGRADE D: pool-relative corrected yaw from lane_heading_node.
        self.lane_yaw = float(msg.data)
        self.lane_yaw_t = self.get_clock().now().nanoseconds * 1e-9

    def on_ground_truth(self, msg):
        # Stonefish odometry: NED world position. Convert to compare frame.
        p = msg.pose.pose.position
        pos_ned = np.array([p.x, p.y, p.z])
        self.gt_x_ned = float(p.x)     # UPGRADE A: feeds the in-node floor profile
        px, py, pz = gt_world_to_compare(pos_ned, self.frame)
        # GT NED yaw (orientation, not velocity-derived) — used ONLY to verify, in the
        # running log, that the GTSAM graph's own attitude tracks truth better than the
        # published (yaw_drift-corrupted) IMU orientation. Never fed into any estimator.
        qo = msg.pose.pose.orientation
        self.gt_yaw_ned = float(_yaw_from_quat_xyzw(qo.x, qo.y, qo.z, qo.w))
        # GT planar velocity (compare frame) for the flow->IMU yaw autocal. Light
        # low-pass; guarded against absurd gaps. Never fed into any estimator — only
        # used to measure the fixed extrinsic yaw offset.
        t_gt = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self._gt_prev_xy is not None and self._gt_prev_t is not None:
            dtg = t_gt - self._gt_prev_t
            if 1e-3 < dtg < 1.0:
                vx = (px - self._gt_prev_xy[0]) / dtg
                vy = (py - self._gt_prev_xy[1]) / dtg
                if self._gt_vel is None:
                    self._gt_vel = (vx, vy)
                else:
                    a = 0.3
                    self._gt_vel = (a * vx + (1 - a) * self._gt_vel[0],
                                    a * vy + (1 - a) * self._gt_vel[1])
        self._gt_prev_xy = (px, py)
        self._gt_prev_t = t_gt
        if self.gt_anchor is None:
            # All estimators dead-reckon from ZERO, but ground truth starts at the spawn
            # (start_position="-12.1 0 0.3" in sauvc_qualification.scn). Anchor every track
            # to ground truth's first pose so the comparison is displacement-from-start,
            # which is what dead reckoning actually measures. Without this the plots carry
            # a fixed ~12.1 m x-offset and can never overlay.
            self.gt_anchor = (px, py, pz)
        self._publish('ground_truth', px, py, pz, msg.header.stamp)

    def on_image(self, msg):
        if self.bridge is None:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        dt = 0.0 if self.prev_img_t is None else (t - self.prev_img_t)
        self.prev_img_t = t

        # gyro in camera frame: down camera x=image right, y=image down.
        # FIX(derotation axes, corrected): my earlier patch rotated body rates by the
        # mount yaw as a pure z-rotation — WRONG, because (image-right, image-down) is
        # not reached from body FLU by a z-rotation alone: the camera z points DOWN
        # while FLU z points UP, which reverses the yaw sense. Derive by dot products
        # with the camera axes instead. For the sim mount (rpy 0 0 1.5708 in FRD):
        #   x_cam = starboard = FLU(0,-1,0)  ->  w_cam_x = -wy
        #   y_cam = aft       = FLU(-1,0,0)  ->  w_cam_y = -wx
        # This is EXACTLY the mapping flow_velocity_node uses on hardware (validated
        # by the Phase 3 hand-push test), and it holds for the sim mount too.
        # Verify anytime with a stationary pitch/roll wiggle: derotated flow ~0.
        # UPGRADE C: derotate with the gyro at the INTER-FRAME MIDPOINT, not the
        # latest sample — removes lag-induced derotation error during turns.
        t_mid = t - 0.5 * dt if dt > 0.0 else t
        wx_i, wy_i, wz_i = self._gyro_at(t_mid)
        gyro_xy_cam = (-wy_i, -wx_i)
        # UPGRADE A+B: in-node camera altitude (profile + datum + tilt). Fallback to
        # the shim /altitude only when self_altitude is explicitly disabled.
        if self.self_alt:
            alt = self._camera_altitude(t)
        else:
            alt = self.altitude if self.altitude is not None else max(
                self.get_parameter('pool_depth').value - self.depth, 0.1)

        res = self.flow.estimate(gray, dt, gyro_xy_cam, alt)
        if res is None:
            # FIX(dropout visibility): every None here is displacement PERMANENTLY
            # lost from the dead-reckoned track (the log showed 48% of GT-moving
            # intervals frozen). Count it and warn, throttled to 1 Hz.
            if dt > 0.0:
                self._drop_count += 1
                self._drop_streak += 1
                if t - self._last_drop_warn > 1.0:
                    self._last_drop_warn = t
                    reason = getattr(self.flow, 'last_failure', None) or 'unknown'
                    counts = getattr(self.flow, 'fail_counts', {})
                    self.get_logger().warn(
                        f'flow dropout #{self._drop_count} '
                        f'(streak {self._drop_streak}, reason: {reason}, '
                        f'totals: {counts}). With the reference-hold fix in '
                        'flow_core the gap is recovered on the next good track '
                        "UNLESS the reason ends in '+track_lost' — those gaps are "
                        'genuinely lost motion (CPU starvation? check rtf_monitor).')
        else:
            self._drop_streak = 0
            # FIX(y mirror): flow_core outputs body FLU ("image right is -y body,
            # since y is left"). Rotating by a NED yaw requires the body vector in
            # FRD; planar FLU->FRD is vy -> -vy (frames.frd_to_flu_vec involution,
            # same map both directions). Without this flip every lateral (sway/east)
            # leg integrates MIRRORED — exactly the GT +2.28 m East vs flow -0.55 m
            # seen at the end of the log. In 'enu' the compare yaw is ENU and the
            # FLU vector is already correct, so no flip.
            vx_b = res['vx']
            vy_b = -res['vy'] if self.frame == 'ned' else res['vy']

            # UPGRADE F: ZUPT — when the flow AND the gyro both read ~zero, the
            # vehicle is stationary; clamp velocity to exactly 0 so integrator noise
            # cannot creep (the +0.09 m drift measured during the dive).
            gyro_mag = math.sqrt(wx_i * wx_i + wy_i * wy_i + wz_i * wz_i)
            stationary = (self.zupt_on
                          and math.hypot(res['vx'], res['vy']) < self.zupt_vel
                          and gyro_mag < self.zupt_gyro)
            if stationary:
                vx_b = vy_b = 0.0
                self._zupt_count += 1

            # UPGRADE D: yaw with optional lane-heading substitution
            yaw_cmp, yaw_ned_used = self._fused_yaws(t)

            # FLOW->IMU YAW ALIGNMENT: measure the fixed offset from GT (sim) while it
            # is un-frozen, then apply it to the yaw used by ALL flow-based estimators.
            # Measured against the RAW yaw so the estimate is independent of what is
            # already applied. This is what stops y from drifting while moving in x.
            if not self._yawcal_frozen:
                self._update_yaw_autocal(vx_b, vy_b, yaw_cmp)
            yaw_cmp_f = yaw_cmp + self.flow_yaw_offset
            yaw_ned_f = yaw_ned_used + self.flow_yaw_offset

            # UPGRADE E: quality-scaled measurement variance (flow_velocity_node's
            # model): worse spread / fewer inliers -> larger R -> less trust.
            r_var = (self.r_flow_base * (1.0 + res['spread_px'])
                     * (100.0 / max(res['n_inliers'], 1)))
            if stationary:
                r_var = 1e-4      # a true zero is a very confident measurement

            # body velocity -> EKF + integrate to position (yaw offset-corrected)
            self.ekf.update_flow(vx_b, vy_b, yaw_cmp_f, t, r_var=r_var)
            fx_, fy_ = self.flow_pos.update(vx_b, vy_b, yaw_cmp_f, t)
            fz = depth_to_world_z(self.depth, self.frame)
            ax, ay, az = self._anchor(fx_, fy_, fz)
            self._publish('flow', ax, ay, az, msg.header.stamp,
                          vx=res['vx'], vy=res['vy'])

            # EKF publish (position + velocity)
            ex, ey, ez = self.ekf.position
            evx, evy = self.ekf.velocity
            aex, aey, aez = self._anchor(ex, ey, ez)
            self._publish('ekf', aex, aey, aez, msg.header.stamp, vx=evx, vy=evy)

            # GTSAM path: independent metric velocity, integrated
            if self.gtsam.available:
                # Rotate flow body velocity into NED world (the graph runs in NED).
                # FIX(y mirror): the graph world is ALWAYS NED regardless of the
                # compare frame, so the FLU->FRD flip is unconditional here.
                vy_frd = -vy_b if self.frame != 'ned' else vy_b   # vy_b is FRD in 'ned'

                # WHICH YAW TO ROTATE BY: confirmed by reading Stonefish's IMU.cpp
                # directly — accumulatedYawDrift (the .scn's yaw_drift param) is added
                # ONLY to the published orientation's yaw channel, as a pure post-hoc
                # ramp; the raw angular_velocity channel this graph's add_imu() consumes
                # is computed from the TRUE angular velocity, entirely upstream of that
                # injection. So the graph's OWN preintegrated attitude never sees the
                # drift at all, while yaw_ned_f (used by the EKF/integrator, which have
                # no other yaw source) is derived from the published orientation and
                # DOES carry it — corrected only by the external EWMA's ~0.05 deg lag.
                # Using yaw_ned_f here too would feed the graph a "measurement" already
                # rotated by a signal its own attitude doesn't need external help with,
                # partly defeating the point of letting it fuse an independent source.
                # Fall back to yaw_ned_f only before the graph has an attitude of its
                # own (i.e. for the very first, initializing sample).
                yaw_for_gtsam = self.gtsam.current_ned_yaw()
                if yaw_for_gtsam is None:
                    yaw_for_gtsam = yaw_ned_f
                c, sn = np.cos(yaw_for_gtsam), np.sin(yaw_for_gtsam)
                fv_ned = np.array([c * vx_b - sn * vy_frd,
                                   sn * vx_b + c * vy_frd, 0.0])
                if not self.gtsam.initialized:
                    # Seed attitude from the AHRS (gravity-aligned) + initial flow
                    # velocity. FIX: only once a real IMU quat exists — initializing
                    # from the identity placeholder mis-aligns gravity.
                    if self._have_imu:
                        q = self.last_quat_wxyz_ned
                        self.gtsam.initialize(q, fv_ned, self.depth)
                elif t - self._last_kf_t >= self.kf_period:
                    # FIX(dropout/CPU): throttle the full iSAM2 update to ~5 Hz
                    # keyframes. add_imu keeps preintegrating at IMU rate in between,
                    # so the factor between keyframes still spans all the motion.
                    self._last_kf_t = t
                    # UPGRADE(parity): weight the graph's flow prior with the SAME
                    # quality-scaled evidence as the EKF (sigma = sqrt(var)), and
                    # near-zero under ZUPT — a confidently-stationary prior is
                    # exactly what pins the graph's velocity bias estimation.
                    # Pass the IMU orientation so the graph gets a loose absolute
                    # attitude anchor. Without it, yaw is unobservable at low speed and
                    # the graph diverges (see gtsam_estimator.add_keyframe). roll/pitch
                    # are gravity-referenced (trustworthy); yaw is loosely weighted.
                    out = self.gtsam.add_keyframe(fv_ned, self.depth,
                                                  imu_quat_wxyz=self.last_quat_wxyz_ned
                                                  if self._have_imu else None)
                    if out is not None:
                        pos_ned, vel_ned = out
                        # graph pos is NED absolute (z=depth). Convert to compare frame,
                        # then anchor x/y to ground truth start like the others.
                        gp = gt_world_to_compare(np.array(pos_ned), self.frame)
                        gx, gy, gz = self._anchor(gp[0], gp[1], gp[2])
                        self._publish('gtsam', gx, gy, gz, msg.header.stamp,
                                      vx=float(vel_ned[0]), vy=float(vel_ned[1]))

        if self.show_cam:
            cv2.imshow('down camera', frame_bgr)
        if self.show_flow:
            cv2.imshow('optical flow', self.flow.overlay(frame_bgr))
        if self.show_cam or self.show_flow:
            cv2.waitKey(1)

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

    def _publish(self, key, x, y, z, stamp, vx=0.0, vy=0.0):
        self._latest[key] = (x, y, z)
        self._maybe_print(stamp)
        od = Odometry()
        od.header.stamp = stamp
        od.header.frame_id = 'map_' + self.frame
        od.child_frame_id = 'base_link'
        od.pose.pose.position.x = float(x)
        od.pose.pose.position.y = float(y)
        od.pose.pose.position.z = float(z)
        od.twist.twist.linear.x = float(vx)
        od.twist.twist.linear.y = float(vy)
        self.pubs[key].publish(od)

    def _maybe_print(self, stamp):
        if not self.print_est:
            return
        t = stamp.sec + stamp.nanosec * 1e-9
        if t - self._last_print < self.print_period:
            return
        self._last_print = t

        def fmt(v):
            if v is None:
                return f"{'--':>8} {'--':>8} {'--':>8}"
            return f"{v[0]:+8.3f} {v[1]:+8.3f} {v[2]:+8.3f}"

        order = ['ground_truth', 'flow', 'ekf', 'pressure', 'gtsam']
        lines = [f"\n─ estimates [{self.frame.upper()} frame]  "
                 f"x        y        z ─────────────"]
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
        print('\n'.join(lines))


def main():
    rclpy.init()
    node = FlowEvalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.show:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
