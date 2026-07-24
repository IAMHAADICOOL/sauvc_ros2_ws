# ADD to scripts/eval_parameters.py, inside declare_eval_parameters(node),
# anywhere after the existing lane-heading block (order does not matter, the
# node reads them by name). These are the YAW-UPGRADE parameters (2026-07-23).
# (zupt_mean_vel / zupt_window from the coast-blindness fix are already in your
# eval_parameters.py — the uploaded node reads them — so they are not repeated.)

        # --- YAW UPGRADE: 7-state EKF (psi + gyro z-bias) + lane fusion ---
        # The EKF's absolute yaw sensor is the RAW stamped line measurement from
        # lane_heading_node, NOT /heading/pool_relative (that topic blends the
        # bias-drifting IMU yaw back in — see flow_eval_node's __init__ comment).
        node.declare_parameter('ekf_use_lane', True)
        node.declare_parameter('lane_meas2_topic', '/heading/line_meas2')
        # base 1-sigma of one line measurement [deg]; the accepted-frame
        # residuals in the 2026-07 logs sat at 1-2 deg.
        node.declare_parameter('ekf_lane_sigma_deg', 2.0)
        # sigma scaled by (1 + slope*(1-R)), R = line concentration in [0.6, 1]:
        # a barely-passing R=0.6 frame carries ~2.2x the base sigma.
        node.declare_parameter('ekf_lane_sigma_r_slope', 3.0)
        # psi process-noise density [rad^2/s]: gyro white noise (~gtsam
        # gyro_sigma 0.0017 -> 3e-6/s) padded for published-yaw quantization.
        node.declare_parameter('ekf_q_yaw', 1.0e-5)
        # b_psi random walk [rad^2/s]: the sim bias is CONSTANT — keep tiny so
        # the estimate converges instead of wandering.
        node.declare_parameter('ekf_q_bias', 1.0e-10)
        # GTSAM attitude-prior yaw sigma [rad] when NO fresh lane measurement
        # exists and the prior can only hold the graph's own yaw (trust-region
        # against the rank-deficiency teleports). Wider than the ctor's 0.2 so
        # long dropouts lean on the (bias-corrected) gyro, not on themselves.
        node.declare_parameter('gtsam_att_yaw_sigma_hold', 0.5)
