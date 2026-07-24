#!/usr/bin/env python3
"""eval_parameters.py — every ROS parameter flow_eval_node declares, in one place.

Split out of flow_eval_node.py: the declaration block was ~140 lines of the
constructor and is almost entirely explanatory comment, which is exactly the kind
of block that is easier to read (and to grep for a parameter name) on its own.

The body below is the ORIGINAL block verbatim — same order, same defaults, same
comments — with `p = self.declare_parameter` rebound to `p = node.declare_parameter`
so it can take the node as an argument. Nothing else changed, so
`ros2 param list` output is identical to before the split.
"""


def declare_eval_parameters(node):
    """Declare every flow_eval_node parameter on `node`. Call once, first thing in
    __init__, before anything reads a parameter value."""
    p = node.declare_parameter
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
    # 'slam' : TRUE SLAM state augmentation (UPGRADE H). EKF: each feature appends
    #          [fx fy] to the state with full cross-covariance (FEKFSLAM math);
    #          gate's rulebook x fused as a one-shot feature-coordinate
    #          measurement, gate y LEARNED at birth and thereafter legitimately
    #          correcting robot y through the correlation. GTSAM: Point3 L(j)
    #          landmarks + BearingRangeFactor3D, anisotropic gate birth prior.
    #          Guards: birth after N consistent sightings, chi2 innovation gate
    #          (EKF), Huber kernels (graph). Assumes compare_frame='ned'.
    p('landmark_mode', 'off')
    p('features_topic', '/vision/features')
    p('gate_x', 4.4)                  # fallback if scene_file wasn't given
    p('lm_min_frames', 5)             # sightings before a landmark is frozen
    p('lm_max_first_range', 8.0)      # m, only near sightings shape the first fix
    p('lm_innov_gate', 2.0)           # m, reject corrections larger than this
    p('lm_obs_sigma', 0.05)           # observation sigma = lm_obs_sigma * range
    p('lm_map_inflate', 1.5)          # stored-variance inflation (decoupling tax)
    # --- UPGRADE H ('slam' mode) observation-noise model + gate prior ---
    p('gate_sigma_x', 0.2)             # rulebook trust in the gate line x [m]
    p('lm_sigma_bearing_deg', 1.7)     # detector bearing 1-sigma [deg]
    p('lm_sigma_range_a', 0.10)        # sigma_r = a + b*r^2 [m] (size-ranging law)
    p('lm_sigma_range_b', 0.02)
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
    # --- UPGRADE I: per-run console log to a timestamped txt file ---
    p('log_to_file', False)            # tee ALL console output (prints + WARNs)
    p('log_dir', '~/flow_eval_logs')   # one flow_eval_YYYYmmdd_HHMMSS.txt per run
    # --- LIVE TRAJECTORY PLOT (module: live_traj_plot.py) ---
    # live_plot:=true opens a top-down x-y window overlaying ground truth
    # and every estimate. The view is DATA-centred, not origin-centred: it
    # starts tight around the first pose (wherever the vehicle spawns in the
    # world frame — no manual offsets needed, _anchor() already put all
    # tracks in one consistent compare-frame coordinate system) and the
    # scale grows monotonically as the tracks spread. On Ctrl+C the current
    # frame is saved into log_dir (same folder as the run log) with a
    # timestamped, collision-suffixed name so nothing is ever overwritten.
    p('live_plot', False)
    p('live_plot_rate', 5.0)           # window refresh [Hz]
    p('live_plot_tracks', 'ground_truth,flow,ekf,eskf,gtsam,dvl,tile_grid')  # 'pressure'
    #   excluded on purpose: its x/y are pinned to 0 and would drag the
    #   autoscaled view out to the world origin.
    p('live_plot_min_span', 2.0)       # [m] never zoom tighter than this
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
    p('flow_yaw_offset', 0.0)          # rad, initial flow-body -> IMU-yaw correction
    p('flow_yaw_autocal', True)        # sim only: estimate offset from GT straight legs
    p('flow_yaw_cal_min_speed', 0.05)  # m/s, only sample when clearly translating
    # EWMA FIX: the offset is NOT fixed — the .scn's yaw_drift is a GROWING random
    # walk, so a freeze-once estimate is correct only at the instant it froze and
    # increasingly stale afterwards. Replaced with a never-freezing EWMA on the
    # unit circle: steady-state lag ~= drift_per_sample / alpha. At yaw_drift
    # 0.00029 rad/s sampled ~30 Hz with alpha 0.02 the lag is ~0.03 deg — bounded
    # forever, instead of an error that grows without bound after a freeze.
    p('flow_yaw_cal_alpha', 0.02)      # EWMA gain per accepted sample
    p('flow_yaw_cal_min_n', 5)         # samples before the estimate is applied
    # --- UPGRADE F: ZUPT ---
    p('zupt', True)
    p('zupt_vel', 0.03)                # m/s: below this AND
    p('zupt_gyro', 0.02)               # rad/s: below this -> clamp v to 0
    
    # FIX(coast blindness, 2026-07-23): the ZUPT stationarity test is now a
        # windowed ZERO-MEAN test, not a per-frame deadband. zupt_mean_vel is the
        # threshold on ||mean(raw flow v)|| over the window: zero-mean LK noise
        # (true standstill) averages to ~sigma/sqrt(N) ~ 0.004 m/s over 1 s @
        # 20 fps, while the post-teleop coast measured 2026-07-23 held a 0.019 m/s
        # bias -- 0.010 sits cleanly between them. zupt_window is the averaging
        # baseline; ZUPT can only ENGAGE once the window is >= 80% full, and it
        # disengages instantly when any condition fails.
    p('zupt_mean_vel', 0.010)   # m/s, windowed-mean gate
    p('zupt_window', 1.0)       # s, averaging baseline
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
    p('flow_compensate_vz', False)
    # --- TILE-GRID estimator (grid-phase odometry on the pool floor tiles) ---
    p('tile_pitch', 0.05)              # grout pitch [m]; uv_mode=2 -> 20 tiles/m
    p('tile_patch', 256)               # analysis patch side [px]
    p('tile_min_quality', 1.5)         # fold-peak z-score gate (coast below)
    # world (NED) angle of the IMAGE X axis minus vehicle NED yaw. my_auv.scn
    # mounts camera_down with rpy z=+90deg -> image right = body right -> +pi/2.
    # Tune like flow's sign_x/sign_y if the mount differs (verify vs GT).
    p('tile_cam_yaw_offset', 1.5708)
# --- YAW UPGRADE: 7-state EKF (psi + gyro z-bias) + lane fusion ---
    # The EKF's absolute yaw sensor is the RAW stamped line measurement from
    # lane_heading_node, NOT /heading/pool_relative (that topic blends the
    # bias-drifting IMU yaw back in — see flow_eval_node's __init__ comment).
    p('ekf_use_lane', True)
    p('lane_meas2_topic', '/heading/line_meas2')
    # base 1-sigma of one line measurement [deg]; the accepted-frame
    # residuals in the 2026-07 logs sat at 1-2 deg.
    p('ekf_lane_sigma_deg', 2.0)
    # sigma scaled by (1 + slope*(1-R)), R = line concentration in [0.6, 1]:
    # a barely-passing R=0.6 frame carries ~2.2x the base sigma.
    p('ekf_lane_sigma_r_slope', 3.0)
    # psi process-noise density [rad^2/s]: gyro white noise (~gtsam
    # gyro_sigma 0.0017 -> 3e-6/s) padded for published-yaw quantization.
    p('ekf_q_yaw', 1.0e-5)
    # b_psi random walk [rad^2/s]: the sim bias is CONSTANT — keep tiny so
    # the estimate converges instead of wandering.
    p('ekf_q_bias', 1.0e-10)
    # GTSAM attitude-prior yaw sigma [rad] when NO fresh lane measurement
    # exists and the prior can only hold the graph's own yaw (trust-region
    # against the rank-deficiency teleports). Wider than the ctor's 0.2 so
    # long dropouts lean on the (bias-corrected) gyro, not on themselves.
    p('gtsam_att_yaw_sigma_hold', 0.5)
    
    p('eskf_enabled', True)
    # PER-SAMPLE noise stddevs at the IMU rate — the values every static
    # init this week has measured for this scene (0.02 / 0.0017 @ 100 Hz).
    # Converted to continuous densities internally (the unit fix).
    p('eskf_accel_sigma', 0.02)
    p('eskf_gyro_sigma', 0.0017)
    # Bias random walks: the plant biases are CONSTANT (modified IMU.cpp),
    # so these are deliberately tighter than the graph's (validation T4:
    # looser values made the bias estimate oscillate enough to lose the
    # lane-dropout coast advantage).
    p('eskf_accel_bias_rw', 1.0e-4)
    p('eskf_gyro_bias_rw', 1.0e-5)
    # Roll/pitch soft anchor sigma [deg] (gravity-referenced AHRS source,
    # same channel the graph's attitude prior trusts).
    p('eskf_rp_sigma_deg', 0.5)
    # Static init: same pattern/reasons as the graph, but OPTIONAL here —
    # a recursive filter converges its biases online regardless, so
    # eskf_init_min_samples:=0 starts instantly from zero biases.
    p('eskf_init_min_samples', 200)
    p('eskf_init_settle_skip_s', 1.0)
