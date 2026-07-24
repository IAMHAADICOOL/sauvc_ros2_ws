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
  /eval/tile_grid      grid-phase odometry on the pool floor tiles (uv_mode="2":
                       20 tiles / 1 m texture = 0.05 m pitch). Absolute position mod
                       one tile every frame -> sub-tile error cannot accumulate.
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
  H. LANDMARK SLAM (landmark_mode='slam'): true FEKFSLAM-style state augmentation in
     the EKF (features appended to the state with cross-covariance; gate's rulebook x
     fused as a one-shot coordinate measurement, gate y LEARNED then legitimately
     correcting robot y) and Point3 landmarks + BearingRangeFactor3D in the GTSAM
     graph. 'off'/'gate'/'map' behave exactly as before. See README_SLAM.md.
"""
# ---------------------------------------------------------------------------
# MODULE LAYOUT (refactor only - no behavior change). What used to be one 1300-line
# file is now this node plus seven modules in scripts/, each holding one coherent
# group. Every comment, docstring and default from the original is preserved
# verbatim in whichever module its code moved to:
#   scripts/eval_run_log.py        RunLogTee (UPGRADE I console tee)
#   scripts/eval_quat_utils.py     quaternion/frame helpers + their verification notes
#   scripts/eval_scene_parser.py   _parse_scene (.scn start pose + landmark map)
#   scripts/eval_parameters.py     every declare_parameter, same order/defaults
#   scripts/eval_helpers_mixin.py  altitude, gyro sync, yaw autocal, lane fusion
#   scripts/eval_landmark_mixin.py /vision/features -> gate/map/slam corrections
#   scripts/eval_publish_mixin.py  anchoring, /eval/* publish, terminal report
# The three mixins are mixed into FlowEvalNode below, so every `self.` reference
# resolves exactly as it did when they were methods in this file.
# ---------------------------------------------------------------------------
import math
import os
import sys
import threading
import time as _time
from collections import deque
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, Image, FluidPressure
from std_msgs.msg import Float32
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import SetParametersResult
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
from sauvc_flow_eval.estimators.eskf_estimator import EskfEstimator
from sauvc_flow_eval.estimators.gtsam_estimator import GtsamEstimator
from sauvc_flow_eval.estimators.tile_grid_estimator import TileGridEstimator
from sauvc_flow_eval.scripts.live_traj_plot import TrajectoryPlotter
from sauvc_sim_bridge.frames import ned_frd_quat_to_enu_flu, flu_frd_to_ned_wxyz
# --- split-out modules (see MODULE LAYOUT above) ---
from sauvc_flow_eval.scripts.eval_run_log import RunLogTee
from sauvc_flow_eval.scripts.eval_quat_utils import (
    _enu_quat_to_ned_wxyz, _quat_to_R, _yaw_from_quat_xyzw,
    _rp_from_quat_wxyz, _quat_wxyz_from_rpy)
from sauvc_flow_eval.scripts.eval_scene_parser import _parse_scene
from sauvc_flow_eval.scripts.eval_parameters import declare_eval_parameters
from sauvc_flow_eval.scripts.eval_helpers_mixin import EvalHelpersMixin
from sauvc_flow_eval.scripts.eval_landmark_mixin import EvalLandmarkMixin
from sauvc_flow_eval.scripts.eval_publish_mixin import EvalPublishMixin


class FlowEvalNode(EvalHelpersMixin, EvalLandmarkMixin, EvalPublishMixin, Node):
    def __init__(self):
        super().__init__('flow_eval_node')
        # Every declare_parameter call now lives in scripts/eval_parameters.py -
        # same order, same defaults, same comments. Must run before the `g` lambda
        # below reads any of them.
        declare_eval_parameters(self)

        g = lambda n: self.get_parameter(n).value
        # --- floor-profile altitude: which estimate's NED x to look up on (runtime) ---
        self._ALT_X_SOURCES = ('ground_truth', 'ekf', 'eskf', 'gtsam', 'flow', 'tile_grid')
        self.declare_parameter('altitude_x_source', 'ground_truth')   # default = old behavior
        self.alt_x_source = g('altitude_x_source')
        if self.alt_x_source not in self._ALT_X_SOURCES:
            raise ValueError(f"altitude_x_source must be one of {self._ALT_X_SOURCES}, "
                            f"got {self.alt_x_source!r}")
        self._alt_xy = {}          # source -> latest ANCHORED (x, y), compare frame
        self._alt_x_last = None    # last good NED x, fallback when a source is silent
        self._alt_warned_src = None
        self.add_on_set_parameters_callback(self._on_set_params)
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
        self.yawcal_alpha = float(g('flow_yaw_cal_alpha'))
        self.yawcal_min_n = int(g('flow_yaw_cal_min_n'))
        self._yawcal_s = 0.0            # EWMA of sin(offset_sample)  (unit circle)
        self._yawcal_c = 0.0            # EWMA of cos(offset_sample)
        self._yawcal_n = 0
        self._yawcal_log_t = 0.0        # throttle for the periodic offset report
        self._gt_vel = None            # ground-truth planar velocity (compare frame)
        self._gt_prev_xy = None
        self._gt_prev_t = None
        self.zupt_on = bool(g('zupt'))
        self.zupt_vel = float(g('zupt_vel'))
        self.zupt_gyro = float(g('zupt_gyro'))
        # FIX(coast blindness): windowed zero-mean stationarity test parameters.
        # zupt_mean_vel separates zero-mean LK noise (true standstill, mean ->
        # ~0.004 m/s over the window) from persistent sub-threshold drift (the
        # ~0.02 m/s post-teleop coast that froze flow/ekf/gtsam for 29 s on
        # 2026-07-23 while tile_grid tracked GT). zupt_window is the averaging
        # baseline in seconds.
        self.zupt_mean_vel = float(g('zupt_mean_vel'))
        self.zupt_window = float(g('zupt_window'))
        self._zupt_buf = deque()                # (t, vx_b, vy_b) raw, pre-clamp
        self._zupt_mean = 0.0
        self._zupt_engaged = False
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
        # UPGRADE H ('slam' mode) state
        self.gate_sigma_x = float(g('gate_sigma_x'))
        self.lm_sig_b = math.radians(float(g('lm_sigma_bearing_deg')))
        self.lm_sig_ra = float(g('lm_sigma_range_a'))
        self.lm_sig_rb = float(g('lm_sigma_range_b'))
        self._gate_prior_set = set()   # gate names whose graph birth prior is placed
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
        # --- YAW UPGRADE (2026-07-23, follows the IMU.cpp bias modification) ---
        # The EKF now carries [psi, b_psi] in its state; the RAW mod-90 line
        # measurement (/heading/line_meas2: image angle + concentration R,
        # stamped) is its absolute yaw sensor, and the SAME cached measurement
        # anchors the GTSAM attitude prior's yaw. The published /heading/
        # pool_relative is deliberately NOT used for either: it is the lane
        # node's own complementary blend of the (now bias-drifting) IMU yaw
        # with the line correction — feeding it to a filter that already
        # integrates the same gyro would count the bias twice, and during the
        # lane node's turn-hardening freezes it degenerates to exactly the
        # biased IMU while still presenting as an absolute measurement.
        self.ekf_use_lane = bool(g('ekf_use_lane'))
        self.ekf_lane_sig = math.radians(float(g('ekf_lane_sigma_deg')))
        self.ekf_lane_slope = float(g('ekf_lane_sigma_r_slope'))
        self.gtsam_att_yaw_hold = float(g('gtsam_att_yaw_sigma_hold'))
        self._lane_meas = None                  # (t, ang, R) latest raw line meas
        # Diagnostic state for lane-heading fusion visibility (see _fused_yaws).
        # Previously NOTHING outside on_image's local scope could see whether lane
        # fusion was even active, let alone whether it was helping — the periodic
        # print's "imu=" column always showed raw self.yaw_ned, never the fused
        # value actually fed to the EKF/flow/GTSAM. That made it impossible to
        # answer "is lane_heading_node fixing anything?" from the logs at all.
        self.yaw_ned_fused = 0.0        # the yaw ACTUALLY used downstream this cycle
        self.lane_active = False        # was lane fusion applied THIS cycle
        self._lane_accept_n = 0         # accepted (fresh + within sanity gate)
        self._lane_reject_stale_n = 0   # rejected: no fresh /heading/pool_relative
        self._lane_reject_gate_n = 0    # rejected: disagreed with IMU by >30 deg
        self._lane_summary_last_t = 0.0
        self._lane_summary_period = 15.0   # s, matches the yaw-autocal log cadence
        # latest published value per source, for the terminal table
        self._latest = {k: None for k in
                        ('ground_truth', 'dvl', 'flow', 'ekf', 'eskf', 'pressure',
                         'gtsam', 'tile_grid')}
        self._gt_R = None          # GT body->NED rotation, for the DVL twist
        self._last_print = 0.0
        # --- estimators ---
        self.flow = FlowEstimator(g('fx'), g('fy'), g('cx'), g('cy'),
                                  use_clahe=g('use_clahe'),
                                  grid_rows=g('feature_grid_rows'),
                                  grid_cols=g('feature_grid_cols'),
                                  compensate_vz=g('flow_compensate_vz'))
        # YAW UPGRADE: lane_sign/lane_grid encode the lane measurement model
        # h = fold90(lane_sign*psi - lane_grid). SIGN CORRECTED (2026-07-23,
        # run 225022): the EMPIRICAL model is  ang == gamma - psi_ned (mod 90),
        # established two independent ways: (1) the run's lane report column
        # carried an error of exactly -2*psi (mod 90) across the full circle
        # (gt -99.4 -> +19.2, gt -11.3 -> +24.3, gt +74.6 -> +31.4, gt -113 ->
        # +48.7 — all within ~2 deg of -2*psi), and (2) an offline replica of
        # the full iSAM2 pipeline reproduces BOTH observed failure modes with
        # the old sign and clean tracking with this one. The earlier derivation
        # leaned on the 360-deg spin test, which ran above the lane node's
        # freeze_rate — it validated the ENU/NED reflection of the PUBLISHED
        # yaw while line adaptation was frozen, so the line-angle sign was
        # never actually exercised (every other run sat at psi ~ 0 mod 90,
        # where both signs are indistinguishable). In h-form: 'ned' ->
        # s=-1, grid=-gamma; 'enu' (psi_ned = pi/2 - psi_enu) -> s=+1,
        # grid=-gamma. gamma != 0 remains UNTESTED on live data (all runs used
        # pool_axis_ned_yaw=0) — verify at the venue before trusting a rotated
        # grid.
        self.ekf = EkfEstimator(q_yaw=float(g('ekf_q_yaw')),
                                q_bias=float(g('ekf_q_bias')),
                                lane_sign=(-1.0 if self.frame == 'ned' else 1.0),
                                lane_grid=-self.pool_axis_ned)
        # CONFIG D (2026-07-24): 15-state ERROR-STATE KF — IMU strapdown +
        # recursive filtering, the third architecture next to the kinematic
        # EKF and the smoothing graph. Runs internally in NED (like the graph);
        # fed the SAME FLU->FRD-converted IMU, the same flow/depth/lane. Its
        # lane model takes grid directly: h = fold90(grid - psi_ned).
        self.eskf_on = bool(g('eskf_enabled'))
        self.eskf = EskfEstimator(
            gravity=g('gravity'),
            accel_sigma=float(g('eskf_accel_sigma')),
            gyro_sigma=float(g('eskf_gyro_sigma')),
            accel_bias_rw=float(g('eskf_accel_bias_rw')),
            gyro_bias_rw=float(g('eskf_gyro_bias_rw')),
            lane_grid=self.pool_axis_ned,
            init_min_samples=int(g('eskf_init_min_samples')),
            init_settle_skip_s=float(g('eskf_init_settle_skip_s')))
        self.eskf_rp_sigma = math.radians(float(g('eskf_rp_sigma_deg')))
        self.gtsam = GtsamEstimator(gravity=g('gravity'), compare_frame=self.frame)
        self.flow_pos = PositionIntegrator()
        self.gtsam_pos = PositionIntegrator()
        self.tile = TileGridEstimator(g('fx'), tile_pitch=g('tile_pitch'),
                                      patch=g('tile_patch'),
                                      min_quality=g('tile_min_quality'))
        self.tile_cam_yaw_offset = float(g('tile_cam_yaw_offset'))
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
                     for k in ('ground_truth', 'dvl', 'flow', 'ekf', 'eskf',
                               'pressure', 'gtsam', 'tile_grid')}
        # --- live trajectory plot (see live_traj_plot.py for the design) ---
        self.traj = None
        if bool(g('live_plot')):
            if not _HAVE_CV:
                self.get_logger().warn('live_plot requested but cv2/cv_bridge '
                                       'unavailable - plot disabled')
            else:
                tracks = tuple(t.strip() for t in
                               str(g('live_plot_tracks')).split(',') if t.strip())
                self.traj = TrajectoryPlotter(
                    frame=self.frame, tracks=tracks,
                    min_span_m=float(g('live_plot_min_span')))
                self.create_timer(1.0 / max(float(g('live_plot_rate')), 0.1),
                                  self._on_traj_timer)
                self.get_logger().info(
                    f'live trajectory plot ON ({", ".join(tracks)})')
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
        if self.ekf_use_lane or self.use_lane:
            # RAW stamped line measurement (angle + concentration R) — the EKF's
            # absolute yaw sensor and the GTSAM attitude prior's yaw anchor.
            self.create_subscription(Vector3Stamped, g('lane_meas2_topic'),
                                     self.on_lane_meas2, 10)
        # DVL truth-velocity row: SIM-ONLY reference (my_auv.scn declares the DVL
        # purely so the flow can be graded; it must NEVER be fused). Import guarded:
        # without stonefish_ros2 msgs (real vehicle) the row simply stays '--'.
        try:
            from stonefish_ros2.msg import DVL as _DVLMsg
            self.create_subscription(_DVLMsg, f'/{robot}/dvl', self.on_dvl,
                                     qos_profile_sensor_data)
            self.get_logger().info('DVL reference row enabled (/%s/dvl)' % robot)
        except ImportError:
            self.get_logger().info('stonefish_ros2 DVL msg unavailable - dvl row off')
        if self.lm_mode in ('gate', 'map', 'slam'):
            from std_msgs.msg import String as _String
            self.create_subscription(_String, g('features_topic'),
                                     self.on_feature, 20)
            if 'GateCenter' in self.landmarks:
                self.gate_x_known = float(self.landmarks['GateCenter'][0])
            self.get_logger().info(
                f"landmark_mode='{self.lm_mode}': gate x = {self.gate_x_known:.2f} "
                f"(from {'scene' if 'GateCenter' in self.landmarks else 'gate_x param'})"
                + (', small-map ON' if self.lm_mode == 'map' else '')
                + (', SLAM state augmentation ON' if self.lm_mode == 'slam' else ''))
            if self.frame != 'ned':
                # The landmark math throughout on_feature rotates body FRD by yaw_ned
                # into NED world and applies it to the compare-frame state — only
                # consistent when compare_frame='ned' (the default and what every
                # landmark run uses). Refusing loudly beats corrupting silently.
                self.get_logger().error(
                    "landmark_mode requires compare_frame='ned' — landmark "
                    "corrections DISABLED for this run.")
                self.lm_mode = 'off'
        if self.show_cam:
            cv2.namedWindow('down camera', cv2.WINDOW_NORMAL)
        if self.show_flow:
            cv2.namedWindow('optical flow', cv2.WINDOW_NORMAL)
        self.get_logger().info(
            f"flow_eval up, compare_frame='{self.frame}'. Publishing /eval/* "
            "(ground_truth, flow, ekf, pressure"
            + (", gtsam" if self.gtsam.available else "") + ", tile_grid). "
            + ("ALTITUDE: self-computed in-node (floor profile at GT x, camera datum, "
               "tilt-compensated); /altitude used only as a cross-check."
               if self.self_alt else "ALTITUDE: from /altitude (shim)."))
        
    def _on_set_params(self, params):
        for prm in params:
            if prm.name == 'altitude_x_source' and prm.value not in self._ALT_X_SOURCES:
                return SetParametersResult(
                    successful=False,
                    reason=f"altitude_x_source must be one of {self._ALT_X_SOURCES}")
            if prm.name == 'altitude_x_source':
                self.alt_x_source = prm.value
        return SetParametersResult(successful=True)

    def _publish(self, name, x, y, *args, **kwargs):
        # Single choke point: every estimator publishes through here, so cache each
        # source's latest ANCHORED (x, y) before delegating to the real publisher.
        self._alt_xy[name] = (x, y)
        return EvalPublishMixin._publish(self, name, x, y, *args, **kwargs)

    def _alt_profile_x(self):
        """NED x the floor profile is evaluated at, chosen by altitude_x_source.
        Anchored tracks are absolute-NED in 'ned' compare frame (so NED x = anchored x)
        and absolute-ENU in 'enu' (NED x = ENU y). Falls back to ground-truth x, then to
        the last good value, so a not-yet-publishing source never crashes the lookup —
        identical to the pre-feature behavior."""
        src = self.alt_x_source
        if src == 'ground_truth':
            x = self.gt_x_ned
        else:
            xy = self._alt_xy.get(src)
            x = None if xy is None else (xy[0] if self.frame == 'ned' else xy[1])
            if x is None and self._alt_warned_src != src:
                self._alt_warned_src = src
                self.get_logger().warn(
                    f"altitude_x_source='{src}' hasn't published yet -> using ground "
                    "truth x meanwhile (is that estimator enabled/available?)")
        if x is None:
            x = self.gt_x_ned          # original behavior
        if x is None:
            x = self._alt_x_last       # GT-absent (hardware): reuse last good
        if x is not None:
            self._alt_x_last = x
        return x
    def on_dvl(self, msg):
        # DVL velocity is body-frame FRD at the sensor (mount rpy 0, lever arm tiny
        # at our rotation rates): rotate to NED with the latest GT attitude, then to
        # the compare frame. Position columns stay '--' - a DVL measures velocity.
        if self._gt_R is None:
            return
        v = msg.velocity
        v_ned = self._gt_R @ np.array([v.x, v.y, v.z])
        v_cmp = gt_world_to_compare(v_ned, self.frame)
        self._latest['dvl'] = (None, None, None, float(v_cmp[0]), float(v_cmp[1]))
        self._maybe_print(msg.header.stamp)
    # ---- sensor callbacks ----
    def on_imu(self, msg):
        # /imu/data is ENU/FLU (imu_shim). CORRECTION (2026-07-23, run 224010):
        # this comment used to claim body accel/gyro are "frame-of-the-body
        # regardless of world convention, so use them directly" for GTSAM. That
        # was WRONG — FLU and FRD differ by y/z sign — and the log proves it two
        # independent ways: (1) static init measured an accel z-"bias" of +19.62,
        # i.e. mean accel +9.81 where NED/FRD expects -9.81 (the 2g frame flip
        # absorbed into the bias state), and (2) the graph's yaw integrated to
        # EXACTLY -gt through the first sustained turn (gt +85.4 / gtsam -84.2,
        # gt +121.3 / gtsam -120.9 ...), because wz_flu = -wz_frd. It stayed
        # invisible before because the old attitude prior re-anchored yaw to the
        # published IMU yaw every keyframe, dragging the graph along against its
        # own mirrored gyro; the lane-anchored prior (correct for bias
        # observability) removed that mask. The FLU->FRD flip for the GTSAM feed
        # happens at the add_imu call below. Everything ELSE keeps the FLU
        # vectors deliberately: the camera derotation mapping (-wy, -wx) was
        # validated by the hand-push test WITH these axes, and ZUPT uses only
        # the gyro magnitude.
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
        # YAW UPGRADE: propagate the EKF's psi with PUBLISHED-YAW INCREMENTS.
        # The modified IMU.cpp guarantees published yaw = integral of the
        # reported (bias-corrupted) z-rate, so wrap(yaw[k]-yaw[k-1]) IS the
        # biased-gyro increment in this node's already-validated convention —
        # no FLU/FRD gyro-sign assumption anywhere (the exact class of silent
        # sign bug this project keeps paying for). The EKF subtracts its own
        # b_psi estimate inside; the published yaw is NEVER fused as absolute
        # (it is the integral of what's already fed — that would double-count
        # the bias).
        self.ekf.propagate_yaw(self.yaw, ti)
        # feed GTSAM preintegration at IMU rate — in body FRD, matching the
        # graph's NED/FRD convention (see the CORRECTION comment above; the
        # flip is frames.frd_to_flu_vec's involution, same map both ways:
        # (x, y, z) -> (x, -y, -z)).
        a = self.accel_body
        w = self.gyro_body
        a_frd = (a[0], -a[1], -a[2])
        w_frd = (w[0], -w[1], -w[2])
        if self.gtsam.available:
            self.gtsam.add_imu(a_frd, w_frd, ti)
        # CONFIG D: the ESKF consumes the identical FRD stream. Attitude seed
        # from the converted (NED/FRD) quat; roll/pitch softly anchored to the
        # SAME gravity-referenced source the graph's attitude prior uses —
        # extracted via _rp_from_quat_wxyz on the NED quat, NOT the raw ENU
        # roll/pitch (FLU->FRD flips pitch). Yaw is never fed from the IMU.
        if self.eskf_on:
            self.eskf.predict(a_frd, w_frd, ti)
            if not self.eskf.initialized:
                self.eskf.try_initialize(self.last_quat_wxyz_ned, self.depth, ti)
            else:
                r_ned, p_ned = _rp_from_quat_wxyz(*self.last_quat_wxyz_ned)
                self.eskf.update_rp(r_ned, p_ned, self.eskf_rp_sigma, ti)
    def on_pressure(self, msg):
        # depth from gauge pressure (rho*g matches the scene; see depth_shim)
        self.depth = msg.fluid_pressure / (1000.0 * 9.81)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pz = depth_to_world_z(self.depth, self.frame)
        self.ekf.update_depth(pz, t)
        if self.eskf_on and self.eskf.initialized:
            self.eskf.update_depth(self.depth, t)    # ESKF is NED-internal
        # /eval/pressure: depth only, x=y=0
        ax, ay, az = self._anchor(0.0, 0.0, pz)
        self._publish('pressure', ax, ay, az, msg.header.stamp)
    def on_altitude(self, msg):
        self.altitude = msg.data
    def on_lane_heading(self, msg):
        # UPGRADE D: pool-relative corrected yaw from lane_heading_node.
        self.lane_yaw = float(msg.data)
        self.lane_yaw_t = self.get_clock().now().nanoseconds * 1e-9

    def on_lane_meas2(self, msg):
        # YAW UPGRADE: one raw line-compass sample (x=image angle in
        # (-pi/4, pi/4], y=concentration R in [0.6, 1], stamped with the image
        # stamp). Cached for the GTSAM attitude prior, and fed to the EKF's
        # mod-90 yaw update with an R-scaled sigma (crisp lines trusted more,
        # same philosophy as the lane node's own gain_r_scaling).
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        ang = float(msg.vector.x)
        R = float(msg.vector.y)
        self._lane_meas = (t, ang, R)
        sig = self.ekf_lane_sig * (1.0 + self.ekf_lane_slope
                                   * max(0.0, 1.0 - R))
        if self.ekf_use_lane:
            self.ekf.update_lane(ang, sig, t)
        if self.eskf_on and self.eskf.initialized:
            self.eskf.update_lane(ang, sig, t)

    def _lane_abs_yaw_ned(self, ref, now_t):
        """Absolute NED yaw from the latest FRESH raw line measurement.
        SIGN CORRECTED (run 225022, see the EkfEstimator construction comment):
        the empirical model is ang == gamma - psi_ned (mod 90), so
        psi_ned == gamma - ang, unwrapped to the branch nearest `ref` (same
        nearest-branch argument as the lane node's own unwrap: valid while ref
        is within 45 deg of truth). The OLD base (ang + gamma) is what made the
        GTSAM attitude prior a mirror-attractor: the offline pipeline replica
        shows the graph following a mirrored prior at exactly rate -1 even
        against a correct gyro, because the flow prior (rotated by the graph's
        own yaw) leaves yaw gauge-free and resisting the prior costs an
        unboundedly growing residual while following it costs a constant one.
        None when stale."""
        if self._lane_meas is None or ref is None:
            return None
        t, ang, _ = self._lane_meas
        if now_t - t > self.lane_fresh_s:
            return None
        base = self.pool_axis_ned - ang
        k = round((ref - base) / (math.pi / 2.0))
        return base + k * (math.pi / 2.0)

    def _yaw_report_line(self):
        """One terminal line: GT yaw plus every source's own yaw estimate (and
        the two bias estimates, in deg/min, against which the .scn yaw_drift is
        directly falsifiable). Errors are vs GT orientation, NED, wrapped."""
        d2 = math.degrees

        def _w(a):
            return (a + math.pi) % (2 * math.pi) - math.pi

        def col(val, err=True):
            if val is None:
                return '    -- '
            e = '' if (not err or self.gt_yaw_ned is None) else \
                f' (err {d2(_w(val - self.gt_yaw_ned)):+6.2f})'
            return f'{d2(_w(val)):+7.2f}{e}'
        ekf_psi, ekf_sig = self.ekf.yaw_est
        ekf_ned = ekf_psi if self.frame == 'ned' else math.pi / 2 - ekf_psi
        ekf_b, _ = self.ekf.yaw_bias
        if self.frame != 'ned':
            ekf_b = -ekf_b       # d(psi_ned)/dt = -d(psi_enu)/dt
        es_yaw = self.eskf.ned_yaw() if (self.eskf_on
                                         and self.eskf.initialized) else None
        es_b = (math.degrees(self.eskf.gyro_bias()[2]) * 60.0
                if es_yaw is not None else None)
        g_yaw = self.gtsam.current_ned_yaw() if self.gtsam.available else None
        gb = self.gtsam.gyro_bias() if self.gtsam.available else None
        now_t = self.prev_img_t if self.prev_img_t is not None else 0.0
        lane_abs = self._lane_abs_yaw_ned(ekf_ned, now_t)
        return ('yaw2[deg]     gt=' + col(self.gt_yaw_ned, err=False)
                + '  imu=' + col(self.yaw_ned)
                + '  ekf=' + col(ekf_ned) + f' s={d2(ekf_sig):4.2f}'
                + '  eskf=' + col(es_yaw)
                + '  gtsam=' + col(g_yaw)
                + '  lane=' + col(lane_abs)
                + f'  | bias[deg/min] ekf={d2(ekf_b) * 60.0:+6.3f}'
                + ('' if es_b is None else f' eskf={es_b:+6.3f}')
                + ('' if gb is None else f' gtsam_bz={d2(gb[2]) * 60.0:+6.3f}')
                + f'  lane_upd ok/gated={self.ekf.lane_ok_n}/'
                + f'{self.ekf.lane_gate_n}')
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
        # twist is in the CHILD (body FRD) frame per the Odometry spec: rotate to
        # NED world with the GT quaternion, then to the compare frame's axes. The
        # exact simulator twist, NOT the differentiated position (that stays
        # dedicated to the yaw autocal's direction measurement above).
        self._gt_R = _quat_to_R(qo.x, qo.y, qo.z, qo.w)
        vb = msg.twist.twist.linear
        v_ned = self._gt_R @ np.array([vb.x, vb.y, vb.z])
        v_cmp = gt_world_to_compare(v_ned, self.frame)
        self._publish('ground_truth', px, py, pz, msg.header.stamp,
                      vx=float(v_cmp[0]), vy=float(v_cmp[1]))
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
        tile_vhint_ned = None
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
            # UPGRADE F (REWORKED): ZUPT with a WINDOWED ZERO-MEAN test, not a
            # per-frame deadband.
            #
            # FIX(coast blindness, 2026-07-23): the old test declared "stationary"
            # whenever the INSTANTANEOUS flow speed dipped below zupt_vel with a
            # quiet gyro. That is a deadband, not a stationarity detector: after
            # teleop stops, the hull keeps drifting at ~0.02 m/s — BELOW the
            # per-frame-pair LK noise floor at this altitude (0.2 px @ 1.4 m,
            # fx=381 -> ~0.015 m/s @ 20 fps), so every single frame read < zupt_vel
            # and the clamp stayed engaged for ~29 s while GT moved 0.53 m
            # (log 211157 blk 293-427; the 0.55-0.6 m final x deficit of flow/ekf/
            # gtsam in ALL THREE trajectory PNGs is exactly this segment).
            # tile_grid was immune because it measures ABSOLUTE grid phase per
            # frame — displacement accumulates in the measurement itself, so a
            # sub-noise-floor VELOCITY is still a perfectly visible POSITION.
            #
            # The discriminator between "true standstill" and "slow creep" is the
            # MEAN of the raw flow velocity over a ~1 s window: zero-mean noise
            # averages toward 0 (sigma/sqrt(N) ~ 0.004 m/s over ~25 frames), while
            # a 0.02 m/s bias stays 0.02. So:
            #   ENTER ZUPT: inst speed < zupt_vel AND gyro < zupt_gyro AND the
            #               window is FULL AND ||mean(v_window)|| < zupt_mean_vel.
            #   EXIT ZUPT:  immediately, the moment any condition fails.
            # Never claim stationary on a part-filled window — absence of evidence
            # is not evidence of standstill. The +0.09 m dive-hold creep the old
            # ZUPT was built for is still killed: at a true hold the windowed mean
            # collapses well below zupt_mean_vel within one window length.
            gyro_mag = math.sqrt(wx_i * wx_i + wy_i * wy_i + wz_i * wz_i)
            # buffer the RAW (pre-clamp) body velocity; gyro gate below guarantees
            # negligible yaw over the window, so a body-frame mean is valid.
            self._zupt_buf.append((t, vx_b, vy_b))
            while self._zupt_buf and t - self._zupt_buf[0][0] > self.zupt_window:
                self._zupt_buf.popleft()
            inst_still = (math.hypot(res['vx'], res['vy']) < self.zupt_vel
                          and gyro_mag < self.zupt_gyro)
            win_full = (len(self._zupt_buf) >= 5
                        and t - self._zupt_buf[0][0] >= 0.8 * self.zupt_window)
            if win_full:
                n = len(self._zupt_buf)
                mvx = sum(b[1] for b in self._zupt_buf) / n
                mvy = sum(b[2] for b in self._zupt_buf) / n
                self._zupt_mean = math.hypot(mvx, mvy)
                win_still = self._zupt_mean < self.zupt_mean_vel
            else:
                win_still = False
            stationary = self.zupt_on and inst_still and win_still
            if stationary:
                vx_b = vy_b = 0.0
                self._zupt_count += 1
                self._zupt_engaged = True
            elif self._zupt_engaged:
                self._zupt_engaged = False
                self.get_logger().info(
                    f'ZUPT disengaged: inst_still={inst_still} '
                    f'win_mean={self._zupt_mean:.3f} m/s '
                    f'(thresh {self.zupt_mean_vel:.3f}) — motion resumed or '
                    'slow creep detected; velocities now pass through raw.')
            # UPGRADE D: yaw with optional lane-heading substitution
            yaw_cmp, yaw_ned_used = self._fused_yaws(t)
            # FLOW->IMU YAW ALIGNMENT: measure the fixed offset from GT (sim) while it
            # is un-frozen, then apply it to the yaw used by ALL flow-based estimators.
            # Measured against the RAW yaw so the estimate is independent of what is
            # already applied. This is what stops y from drifting while moving in x.
            if self.flow_yaw_autocal:
                self._update_yaw_autocal(vx_b, vy_b, yaw_cmp, t)
            yaw_cmp_f = yaw_cmp + self.flow_yaw_offset
            yaw_ned_f = yaw_ned_used + self.flow_yaw_offset
            _vyf = -vy_b if self.frame != 'ned' else vy_b
            _cN, _sN = math.cos(yaw_ned_f), math.sin(yaw_ned_f)
            tile_vhint_ned = (_cN * vx_b - _sN * _vyf, _sN * vx_b + _cN * _vyf)
            # UPGRADE E: quality-scaled measurement variance (flow_velocity_node's
            # model): worse spread / fewer inliers -> larger R -> less trust.
            r_var = (self.r_flow_base * (1.0 + res['spread_px'])
                     * (100.0 / max(res['n_inliers'], 1)))
            if stationary:
                r_var = 1e-4      # a true zero is a very confident measurement
            # body velocity -> EKF (YAW UPGRADE: the filter rotates by ITS OWN
            # psi state — the external yaw_cmp_f path is gone for the EKF; the
            # d/dpsi Jacobian columns this creates are what couple heading to
            # velocity and let lane/landmark evidence correct both coherently)
            self.ekf.update_flow(vx_b, vy_b, t, r_var=r_var)
            # CONFIG D: the ESKF's world is ALWAYS NED, so it needs body FRD
            # unconditionally (same reasoning as the graph's vy_frd below).
            if self.eskf_on and self.eskf.initialized:
                vy_frd_e = -vy_b if self.frame != 'ned' else vy_b
                self.eskf.update_flow(vx_b, vy_frd_e, t, r_var=r_var)
            fx_, fy_ = self.flow_pos.update(vx_b, vy_b, yaw_cmp_f, t)
            fz = depth_to_world_z(self.depth, self.frame)
            ax, ay, az = self._anchor(fx_, fy_, fz)
            # velocity column in the WORLD compare frame like every other row
            # (was body-frame - not comparable), rotated by the SAME offset-corrected
            # yaw the integrator uses.
            cwf, swf = math.cos(yaw_cmp_f), math.sin(yaw_cmp_f)
            self._publish('flow', ax, ay, az, msg.header.stamp,
                          vx=cwf * vx_b - swf * vy_b, vy=swf * vx_b + cwf * vy_b)
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
                # WHICH YAW TO ROTATE BY (REVISED for the modified IMU.cpp).
                # The earlier verified claim — that the gyro channel was clean and
                # only the published orientation carried yaw_drift — is NO LONGER
                # TRUE: IMU.cpp now injects yaw_drift as a constant z-rate BIAS
                # into the reported angular velocity, and the published yaw is its
                # integral (a consistent magnetometer-less AHRS). So the graph's
                # own preintegrated attitude DOES drift now, at the bias rate,
                # until the CombinedImuFactor's bias state converges — which it
                # can, because (a) static init measures the bias directly at spawn
                # (the pre-init gyro mean now CONTAINS it) and (b) the attitude
                # prior below is anchored to the lane compass, giving the graph an
                # absolute yaw reference against which bz becomes observable.
                # Rotating the flow measurement by the graph's OWN yaw remains the
                # right choice: it is the standard VIO structure (measurement
                # model evaluated at the current estimate), stays self-consistent
                # as the bias estimate converges, and still avoids pre-rotating
                # the measurement by an external signal carrying the very error
                # the graph is trying to estimate.
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
                    # ATTITUDE PRIOR YAW (REVISED for the modified IMU.cpp). The
                    # offline finding still stands and now matters MORE: a loose
                    # per-keyframe prior compounds into a tight anchor over
                    # hundreds of keyframes, so whatever yaw goes in here is what
                    # the graph converges to. Under the new IMU model the raw
                    # published yaw is the integral of the SAME biased gyro the
                    # CombinedImuFactor integrates — anchoring to it would make
                    # prior and process agree perfectly while both drift, and bz
                    # would sit at zero with nothing to observe it against (the
                    # silent-failure trap). Therefore:
                    #   * roll/pitch: still from the raw IMU quat — gravity-
                    #     referenced, trustworthy, unaffected by the z-bias.
                    #   * yaw, lane FRESH: the raw line measurement unwrapped to
                    #     the branch nearest the graph's own yaw — a drift-free
                    #     absolute reference, independent of the gyro, which is
                    #     exactly what makes bz observable. Normal (ctor) sigma.
                    #   * yaw, lane STALE: the graph's OWN current yaw with the
                    #     wider gtsam_att_yaw_sigma_hold — a pure trust-region
                    #     term that keeps the near-null yaw direction from the
                    #     rank-deficiency teleports (the +61/-44/-133 deg
                    #     divergence) without injecting any external yaw claim,
                    #     so bias-driven drift during dropouts shows honestly
                    #     instead of being masked.
                    roll_raw, pitch_raw = _rp_from_quat_wxyz(*self.last_quat_wxyz_ned)
                    own_yaw = self.gtsam.current_ned_yaw()
                    lane_abs = self._lane_abs_yaw_ned(
                        own_yaw if own_yaw is not None else yaw_ned_f, t)
                    if lane_abs is not None:
                        att_quat = _quat_wxyz_from_rpy(roll_raw, pitch_raw,
                                                       lane_abs)
                        att_yaw_sig = None            # ctor att_yaw_sigma
                    else:
                        hold = own_yaw if own_yaw is not None else yaw_ned_f
                        att_quat = _quat_wxyz_from_rpy(roll_raw, pitch_raw, hold)
                        att_yaw_sig = self.gtsam_att_yaw_hold
                    out = self.gtsam.add_keyframe(fv_ned, self.depth,
                                                  imu_quat_wxyz=att_quat
                                                  if self._have_imu else None,
                                                  flow_sigma=math.sqrt(r_var),
                                                  att_yaw_sigma=att_yaw_sig)
                    if out is not None:
                        pos_ned, vel_ned = out
                        # graph pos is NED absolute (z=depth). Convert to compare frame,
                        # then anchor x/y to ground truth start like the others.
                        gp = gt_world_to_compare(np.array(pos_ned), self.frame)
                        gx, gy, gz = self._anchor(gp[0], gp[1], gp[2])
                        gv = gt_world_to_compare(np.asarray(vel_ned, float),
                                                 self.frame)
                        self._publish('gtsam', gx, gy, gz, msg.header.stamp,
                                      vx=float(gv[0]), vy=float(gv[1]))
        # ---- TILE-GRID estimator: absolute grid phase every frame ----
        # Runs whether or not flow succeeded; flow only hints the tile unwrap /
        # coasts dropouts. Output is NED-world displacement-from-start.
        tr = self.tile.estimate(gray, t, alt,
                                self.yaw_ned + self.tile_cam_yaw_offset,
                                tile_vhint_ned)
        if tr is not None:
            tp = gt_world_to_compare(np.array([tr['x'], tr['y'], 0.0]), self.frame)
            tzw = depth_to_world_z(self.depth, self.frame)
            tax, tay, taz = self._anchor(tp[0], tp[1], tzw)
            tvx = tvy = None
            if tile_vhint_ned is not None:
                tv = gt_world_to_compare(
                    np.array([tile_vhint_ned[0], tile_vhint_ned[1], 0.0]), self.frame)
                tvx, tvy = float(tv[0]), float(tv[1])
            self._publish('tile_grid', tax, tay, taz, msg.header.stamp,
                          vx=tvx, vy=tvy)
        # CONFIG D publish: OUTSIDE the flow-success block on purpose — the
        # strapdown keeps integrating IMU through flow dropouts, and that
        # coasting behavior is precisely what this architecture demonstrates.
        if self.eskf_on and self.eskf.initialized:
            ep_ = gt_world_to_compare(self.eskf.position_ned, self.frame)
            ev_ = gt_world_to_compare(self.eskf.velocity_ned, self.frame)
            ex_, ey_, ez_ = self._anchor(ep_[0], ep_[1], ep_[2])
            self._publish('eskf', ex_, ey_, ez_, msg.header.stamp,
                          vx=float(ev_[0]), vy=float(ev_[1]))
        if self.show_cam:
            cv2.imshow('down camera', frame_bgr)
        if self.show_flow:
            cv2.imshow('optical flow', self.flow.overlay(frame_bgr))
        if self.show_cam or self.show_flow:
            cv2.waitKey(1)
def main():
    rclpy.init()
    node = FlowEvalNode()
    # UPGRADE I: install the run-log tee right after construction so every print
    # and every get_logger() line from here on lands in the file. Installed here
    # (not inside __init__) so close() is guaranteed by main's finally even if
    # the node dies mid-callback. The file is written incrementally, so Ctrl+C
    # never loses data — close() only restores the fds and appends the footer.
    run_log = None
    if bool(node.get_parameter('log_to_file').value):
        run_log = RunLogTee(str(node.get_parameter('log_dir').value))
        node.get_logger().info(f'run log: teeing all console output to {run_log.path}')
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Save the trajectory frame FIRST, before any window teardown, so the
        # PNG reflects exactly what was on screen at the moment of Ctrl+C. It
        # goes into log_dir — the SAME folder as the run-log txt — with a
        # timestamped name plus _1/_2/... on collision, so no overwrite ever.
        if node.traj is not None:
            saved = node.traj.save(str(node.get_parameter('log_dir').value))
            if saved:
                print(f'[flow_eval_node] trajectory frame saved: {saved}')
        if node.show or node.traj is not None:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()
        if run_log is not None:
            run_log.close()
            print(f'[flow_eval_node] run log saved: {run_log.path}')
if __name__ == '__main__':
    main()
