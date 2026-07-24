#!/usr/bin/env python3
"""estimators/gtsam_estimator.py — iSAM2 visual-inertial factor graph (config B).
REPLACES the earlier version, which misused preintegration (predict-then-reset, no
optimization, no gravity alignment) and diverged to +57 m. That was only HALF of Forster
et al. (RSS 2015): they preintegrate IMU BETWEEN keyframes but OPTIMIZE pos/vel/bias AT
each keyframe against other factors. Preintegration is the motion MODEL; the estimate
comes from the graph. Drop the graph -> open-loop integration of a wrongly-initialized
state, which is exactly what diverged.
Config B (fairest EKF competitor — same three inputs as the eval EKF):
  * CombinedImuFactor       — IMU preintegration + bias evolution in one factor.
  * PriorFactorVector on V  — optical-flow body velocity (rotated to NED).
  * GPSFactor on X (z only) — pressure depth, x/y sigma ~inf so only depth is constrained.
Validated offline before shipping:
  IMU+flow+pressure -> 5 mm error / 6 s with real accel bias
  IMU+pressure only -> 151 mm (no horizontal measurement: drifts in x/y, correctly)
  IMU only          -> 907 mm (bounded now gravity is aligned; was +57 m before)
Two bugs fixed explicitly:
  1. GRAVITY ALIGNMENT. MakeSharedD(g) => n_gravity=(0,0,+g) NED (+z down). Initial Pose3
     attitude MUST be gravity-aligned or ~9.81 m/s^2 specific force integrates as motion.
     We seed attitude from the IMU's fused orientation (AHRS gives it directly).
  2. KEYFRAME INTERVAL = the camera-rate keyframe period (~15 IMU samples). Bounded below
     by bias observability, above by the first-order bias-update linearization validity
     (Forster sec. IV). Only works WITH the optimization between keyframes.
Optional + graceful: gtsam missing -> available=False, eval node skips this estimator.

LANDMARK SLAM (additive; landmark_mode='slam' in flow_eval_node):
  Every named feature from /vision/features becomes a Point3 variable L(j) observed by
  BearingRangeFactor3D from keyframe poses — the 3D twin of the reference 2D
  BearingRangeFactor2D + ISAM2 code, with the CombinedImuFactor as the odometry factor.
  "State expansion" in a factor graph = inserting the new key with an initial value;
  iSAM2 maintains the full joint pose-landmark covariance.
  Isolation guarantee: landmark factors go in a SECOND isam.update() AFTER the VIO
  keyframe update, wrapped in try/except — a degenerate landmark can only lose itself,
  never the VIO chain. Guards: range window, min_sightings-before-birth (median init),
  Huber robust kernels, weak stabilizing prior at birth (or the rulebook's anisotropic
  gate-x prior via set_landmark_prior). The existing add_landmark_xy (gate/map modes)
  is untouched.
"""
import numpy as np
try:
    import gtsam
    from gtsam.symbol_shorthand import X, V, B, L
    _GTSAM = True
except Exception:
    _GTSAM = False
class GtsamEstimator:
    available = _GTSAM
    def __init__(self, gravity=9.81, accel_sigma=0.02, gyro_sigma=0.0017,
                 accel_bias_rw=0.001, gyro_bias_rw=0.0001,
                 flow_sigma=0.05, depth_sigma=0.002, compare_frame='ned',
                 att_prior=True, att_rp_sigma=0.01, att_yaw_sigma=0.2,
                 lm_bearing_sigma=0.03, lm_range_sigma=(0.10, 0.02),
                 lm_min_sightings=3, lm_min_range=0.3, lm_max_range=25.0,
                 lm_default_prior_sigma=5.0, lm_huber_k=1.345,
                 init_min_static_samples=300, init_max_wait_s=6.0,
                 init_settle_skip_s=1.5, imu_rate_hz=100.0):
        self.ok = _GTSAM
        self.frame = compare_frame
        if not self.ok:
            return
        # NOISE UNITS + biasAccOmegaInt (2026-07-23, the ROOT CAUSE of the
        # bias-never-converges saga — found with a single-factor microscope):
        #   1. gtsam's set*Covariance expect CONTINUOUS-TIME noise densities;
        #      accel_sigma/gyro_sigma here are PER-SAMPLE stddevs (what the
        #      static buffer measures, what a datasheet quotes at a rate), so
        #      convert: sigma_cont = sigma_sample * sqrt(1/imu_rate_hz).
        #      Passing per-sample values straight in de-weights every IMU
        #      factor 100x at 100 Hz.
        #   2. setBiasAccOmegaInit: gtsam's DEFAULT is the 6x6 IDENTITY —
        #      i.e. "the linearization-point bias may be wrong by +/-1 rad/s"
        #      — which inflates a 0.2 s preintegrated-rotation sigma to 0.45
        #      rad (26 deg). Under that weighting a 1 deg/min bias leaves a
        #      1e-4-sigma residual per keyframe: bz was UNOBSERVABLE to batch
        #      LM, to forced relinearization, to everything — measured, not
        #      assumed (factor.error identical to 4 decimals for bz=0 vs
        #      bz=true under the default). Set it to a small, honest value:
        #      the factor already carries bias connectivity explicitly, so
        #      this term only covers higher-order linearization error.
        #      With both fixes the per-keyframe bias SNR goes from ~1e-4 to
        #      ~0.3-0.8 sigma and the run total to >10 sigma: ONLINE bias
        #      convergence through the CombinedImuFactor — the original
        #      design intent of config B — actually works now.
        self.imu_rate_hz = float(imu_rate_hz)
        sdt = 1.0 / np.sqrt(self.imu_rate_hz)
        params = gtsam.PreintegrationCombinedParams.MakeSharedD(gravity)
        params.setAccelerometerCovariance(np.eye(3) * (accel_sigma * sdt) ** 2)
        params.setGyroscopeCovariance(np.eye(3) * (gyro_sigma * sdt) ** 2)
        params.setIntegrationCovariance(np.eye(3) * 1e-8)
        params.setBiasAccCovariance(np.eye(3) * accel_bias_rw ** 2)
        params.setBiasOmegaCovariance(np.eye(3) * gyro_bias_rw ** 2)
        params.setBiasAccOmegaInit(np.eye(6) * 1e-6)
        self.params = params
        # FLOW velocity prior: horizontal only. The down-camera optical flow measures
        # HORIZONTAL body translation; vertical motion changes altitude/scale, not
        # horizontal flow, so the flow node reports vz=0 by construction. Constraining
        # world vz to 0 with the same tight sigma (the old Isotropic.Sigma(3, .)) fights
        # the depth factor during any descent/ascent and injects pitch error. Huge vz
        # sigma leaves vertical velocity to the depth factor + IMU, where it belongs.
        self.flow_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([flow_sigma, flow_sigma, 1e6]))
        self.depth_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e6, 1e6, depth_sigma]))
        # ATTITUDE PRIOR (fixes the "gtsam yaw goes wild" divergence). At (near-)zero
        # velocity — the mission's startup hold, and to a lesser degree any straight-line
        # transit — YAW IS UNOBSERVABLE from IMU + a velocity prior + depth: a zero (or
        # forward-only) velocity vector rotated by any heading is equally consistent, so
        # the yaw DOF's information is rank-deficient. Verified offline: the yaw marginal
        # std balloons to ~160 deg (i.e. unconstrained), and iSAM2's Gauss-Newton steps
        # in that near-null direction jump arbitrarily — the +61/-44/-133 deg teleports
        # in the log. A loose absolute attitude prior from the IMU's own orientation
        # removes the rank deficiency (marginal std collapses to the prior sigma). Roll/
        # pitch are gravity-referenced and trustworthy -> tight sigma; yaw is loose
        # (att_yaw_sigma, ~11 deg default) so it only prevents the blow-up, letting
        # real motion/landmarks still refine heading. NOTE(2026-07-23, modified
        # IMU.cpp): repeated weak priors COMPOUND into a tight anchor, so the yaw
        # SOURCE fed into add_keyframe decides what the graph converges to; the
        # node now supplies lane-derived yaw when fresh (making the gyro z-bias
        # observable) and the graph's own yaw with att_yaw_sigma override when
        # stale (pure trust-region). Disable with att_prior=False to reproduce
        # the divergence.
        self.att_prior = bool(att_prior)
        self.att_rp_sigma = float(att_rp_sigma)
        self.att_yaw_sigma = float(att_yaw_sigma)
        self.att_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(
            [att_rp_sigma, att_rp_sigma, att_yaw_sigma, 1e6, 1e6, 1e6])) if self.ok else None
        # iSAM2 with DEFAULT relinearization. REVERTED: an earlier experiment forced
        # relinearizeSkip=1 + a tight threshold to chase periodic print-time spikes. On
        # this CombinedImuFactor graph that made iSAM2 relinearize the IMU factors far
        # from their linearization point every step; the Gauss-Newton steps overshot and
        # the estimate diverged to arbitrarily large values (the "went wild" run). The
        # periodic frame drops it was aimed at are already handled correctly UPSTREAM by
        # throttling keyframes (flow_eval_node's gtsam_keyframe_period ~5 Hz), which is
        # the right lever — not hammering relinearization. Defaults are what the offline
        # validation (5 mm / 6 s) used and what the current clean log reflects.
        self.isam = gtsam.ISAM2()
        self.bias = gtsam.imuBias.ConstantBias()
        self.pim = gtsam.PreintegratedCombinedMeasurements(self.params, self.bias)
        # UPGRADE(Forster TRO16 static initialization): buffer stationary IMU samples
        # BEFORE initialize() so the initial gyro/accel biases can be MEASURED instead
        # of assumed zero. The old tight zero-centred bias prior (sigma 1e-3) fought
        # any real bias exactly like a residual-gravity error; centring the prior on
        # the measured value removes that tension - a large share of the paper's
        # "careful initialization" advantage.
        # WHY THIS IS THE *ONLY* BIAS MECHANISM THAT WORKS HERE (2026-07-23, run
        # 230541 + offline replica): with iSAM2 at DEFAULT relinearization, online
        # bz convergence is effectively STALLED at this problem's evidence scale —
        # the per-keyframe residual a 1 deg/min bias leaves against a lane-anchored
        # attitude prior is ~0.003 deg vs an 11.5 deg prior sigma, and incremental
        # updates never consolidate it across history: the exact-pipeline replica
        # moved bz by 0.001 deg/min in 300 s against a true 1.00 deg/min (a tighter
        # 0.05-rad att sigma changed nothing; forcing relinearizeSkip=1 is the
        # known-divergent lever, reverted earlier). A prior CENTRED on the measured
        # bias, by contrast, is exact from t=0 and the random walk holds it. So the
        # static measurement must actually happen — which is why initialize() now
        # DEFERS (below) until the buffer is genuinely full instead of firing on
        # whatever handful of samples beat the first camera frame (runs collected
        # 28/21/10 samples; even 20-30 is noise-dominated: sigma_gyro/sqrt(N) is
        # ~0.5-1 deg/min against a ~1 deg/min signal).
        # init_min_static_samples: don't initialize before this many pre-init IMU
        #   samples (300 @ 100 Hz = 3 s; the vehicle floats at spawn). More static
        #   time = a better bz: sigma shrinks as 1/sqrt(N).
        # init_max_wait_s: hardware escape hatch — a moving start can never fill
        #   the buffer, so after this much waiting initialize anyway (>=20 samples
        #   still triggers measurement; fewer falls back to the zero-centred prior).
        # init_settle_skip_s: discard the FIRST seconds of samples entirely.
        #   The buffer used to start at t0 — the splash/settle-richest segment —
        #   and the mean of true omega_z over the window enters the "bias"
        #   measurement 1:1: a net yaw of only 0.014 deg across 3 s explains the
        #   whole +0.26 deg/min offset run 231739 measured vs the .scn's 2.9e-4.
        self._init_buf = []          # (accel_xyz, gyro_xyz) while not initialized
        self.init_min_static_samples = int(init_min_static_samples)
        self.init_max_wait_s = float(init_max_wait_s)
        self.init_settle_skip_s = float(init_settle_skip_s)
        self._init_t0 = None
        self._init_buf_max = max(300, self.init_min_static_samples)
        self._init_first_attempt_t = None
        self._init_defer_logged = False
        self.gravity = float(gravity)
        self.k = 0
        self.initialized = False
        self.pose = None
        self.vel = None
        self.t_prev_imu = None
        self._pending_imu = 0
        # --- LANDMARK SLAM bookkeeping (additive; only used when the node feeds
        # add_landmark_obs, i.e. landmark_mode='slam') ---
        self.lm_bearing_sigma = float(lm_bearing_sigma)
        self.lm_range_sigma = tuple(lm_range_sigma)      # sigma_r = a + b*r^2 [m]
        self.lm_min_sightings = int(lm_min_sightings)
        self.lm_min_range = float(lm_min_range)
        self.lm_max_range = float(lm_max_range)
        self.lm_default_prior_sigma = float(lm_default_prior_sigma)
        self.lm_huber_k = float(lm_huber_k)
        self._lm_ids = {}            # name -> j
        self._lm_hist = {}           # name -> list of body-FRD rel obs (pre-birth)
        self._lm_frame_obs = {}      # name -> latest rel obs since last keyframe
        self._lm_priors = {}         # name -> ([val|None]*3, [sigma]*3)
        self._lm_dropped = 0         # landmark updates lost to the try/except shield
    def add_imu(self, accel_xyz, gyro_xyz, t):
        if not self.ok:
            return
        if self.t_prev_imu is None:
            self.t_prev_imu = t
            return
        dt = t - self.t_prev_imu
        self.t_prev_imu = t
        if not self.initialized:
            # UPGRADE(static init): record pre-init samples for bias estimation.
            # SETTLE SKIP: don't buffer until init_settle_skip_s after the first
            # IMU sample — the spawn splash/settle motion otherwise enters the
            # gyro mean 1:1 as fake bias (see __init__).
            if self._init_t0 is None:
                self._init_t0 = t
            if (t - self._init_t0 >= self.init_settle_skip_s
                    and len(self._init_buf) < self._init_buf_max):
                self._init_buf.append((np.asarray(accel_xyz, float),
                                       np.asarray(gyro_xyz, float)))
            return
        if 0.0 < dt < 0.1 and self.initialized:
            self.pim.integrateMeasurement(np.asarray(accel_xyz, float),
                                          np.asarray(gyro_xyz, float), dt)
            self._pending_imu += 1
    def initialize(self, quat_wxyz_ned, init_vel_ned, depth):
        if not self.ok or self.initialized:
            return
        # DEFERRAL (see the static-init comment in __init__): the node retries
        # this every camera frame, so refusing here just postpones init to the
        # next frame. Refuse while the static buffer is below
        # init_min_static_samples, unless init_max_wait_s has elapsed since the
        # first attempt (timed off IMU stamps: robust to sim-time weirdness).
        # Without this gate, whichever handful of IMU samples happened to beat
        # the first camera frame (10 in run 230541) became the "measurement" —
        # or the measurement was skipped outright and bz stayed pinned at the
        # zero-centred prior forever (the online path cannot recover it; see
        # __init__).
        if len(self._init_buf) < self.init_min_static_samples:
            now = self.t_prev_imu
            if self._init_first_attempt_t is None:
                self._init_first_attempt_t = now
            waited = 0.0 if (now is None or self._init_first_attempt_t is None) \
                else now - self._init_first_attempt_t
            if waited < self.init_max_wait_s:
                if not self._init_defer_logged:
                    self._init_defer_logged = True
                    print(f'[gtsam] init DEFERRED: {len(self._init_buf)}/'
                          f'{self.init_min_static_samples} static samples '
                          f'buffered; waiting up to {self.init_max_wait_s:.0f} s '
                          f'(vehicle should be floating still) so the gyro '
                          f'z-bias measurement is not noise-dominated.')
                return
            print(f'[gtsam] init wait EXPIRED after {waited:.1f} s with '
                  f'{len(self._init_buf)} samples — proceeding (moving start?). '
                  f'bz measurement quality degrades as 1/sqrt(N).')
        rot = gtsam.Rot3.Quaternion(*quat_wxyz_ned)
        pos = np.array([0.0, 0.0, float(depth)])
        self.pose = gtsam.Pose3(rot, pos)
        self.vel = np.asarray(init_vel_ned, float)
        # UPGRADE(Forster TRO16 static init): estimate initial biases from the
        # buffered stationary samples. Static specific force in body = R^T (0,0,-g)
        # (NED, gravity +z down); anything beyond that in the accel mean is bias.
        # The gyro mean IS the gyro bias (vehicle floats motionless at spawn). The
        # bias PRIOR is then centred on the measurement with an honest sigma
        # (accel 0.05, gyro 0.005) instead of pinned at zero with sigma 1e-3.
        if len(self._init_buf) >= 20:
            A = np.array([a for a, _ in self._init_buf])
            W = np.array([w for _, w in self._init_buf])
            # ROBUST TRIM: drop samples > 4*MAD from the per-axis median before
            # averaging — a residual settle kick or a single teleop blip is a
            # heavy-tailed contaminant of exactly the mean we are measuring.
            med = np.median(W, axis=0)
            mad = np.median(np.abs(W - med), axis=0) * 1.4826 + 1e-12
            keep = np.all(np.abs(W - med) < 4.0 * mad, axis=1)
            Wk = W[keep] if keep.sum() >= 20 else W
            Ak = A[keep] if keep.sum() >= 20 else A
            acc = Ak.mean(axis=0)
            gyr = Wk.mean(axis=0)
            gyr_sig = Wk.std(axis=0, ddof=1) / max(np.sqrt(len(Wk)), 1.0)
            # AUTO-CALIBRATED noise densities: the same static window gives the
            # actual per-sample noise of THIS sensor/scene (sim .scn or real
            # HFI-A9) — use it instead of the ctor guesses, converted to
            # continuous density. Floors guard against a too-quiet window.
            sdt = 1.0 / np.sqrt(self.imu_rate_hz)
            g_d = max(float(np.mean(Wk.std(axis=0, ddof=1))), 1e-5)
            a_d = max(float(np.mean(Ak.std(axis=0, ddof=1))), 1e-4)
            self.params.setGyroscopeCovariance(np.eye(3) * (g_d * sdt) ** 2)
            self.params.setAccelerometerCovariance(np.eye(3) * (a_d * sdt) ** 2)
            f_expected = rot.matrix().T @ np.array([0.0, 0.0, -self.gravity])
            self.bias = gtsam.imuBias.ConstantBias(acc - f_expected, gyr)
            self.pim = gtsam.PreintegratedCombinedMeasurements(self.params, self.bias)
            bias_noise = gtsam.noiseModel.Diagonal.Sigmas(
                np.array([0.05, 0.05, 0.05, 0.005, 0.005, 0.005]))
            print(f'[gtsam] noise auto-cal: gyro {g_d:.5f}, accel {a_d:.4f} '
                  f'rad/s|m/s^2 per-sample at {self.imu_rate_hz:.0f} Hz '
                  f'-> densities {g_d*sdt:.2e}/{a_d*sdt:.2e}')
            print(f'[gtsam] static init from {len(Wk)}/{len(W)} samples '
                  f'(settle skip {self.init_settle_skip_s:.1f} s, 4-MAD trim): '
                  f'accel_bias={np.round(acc - f_expected, 4)} '
                  f'gyro_bias={np.round(gyr, 5)} '
                  f'+/- {np.round(gyr_sig, 5)} (1-sigma of the mean; bz '
                  f'{np.degrees(gyr[2])*60:+.2f} +/- {np.degrees(gyr_sig[2])*60:.2f} '
                  f'deg/min — this floor, not iSAM2, bounds how close bz can '
                  f'land to the .scn value; it shrinks as 1/sqrt(N))')
        else:
            bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
            print(f'[gtsam] static init skipped ({len(self._init_buf)} < 20 samples)')
        self._init_buf = []
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1]))
        vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        graph.add(gtsam.PriorFactorPose3(X(0), self.pose, pose_noise))
        graph.add(gtsam.PriorFactorVector(V(0), self.vel, vel_noise))
        graph.add(gtsam.PriorFactorConstantBias(B(0), self.bias, bias_noise))
        values.insert(X(0), self.pose)
        values.insert(V(0), self.vel)
        values.insert(B(0), self.bias)
        self.isam.update(graph, values)
        self.initialized = True
    def add_keyframe(self, flow_vel_ned, depth, imu_quat_wxyz=None,
                     flow_sigma=None, att_yaw_sigma=None):
        """flow_sigma: optional per-keyframe stddev [m/s] for the flow velocity
        prior. Parity with the EKF: it scales its flow measurement noise by frame
        quality (spread/n_inliers) and drops it to near-zero under ZUPT; the graph
        should weight the very same evidence the same way. z sigma stays 1e6 ALWAYS
        (flow says nothing about vz). None -> the constructor default.
        att_yaw_sigma: optional per-keyframe yaw stddev [rad] for the attitude
        prior; roll/pitch keep the constructor's att_rp_sigma. The node uses the
        constructor default (None) when the prior's yaw comes from a fresh lane
        measurement, and a wider hold sigma when it can only pass the graph's own
        yaw back as a trust-region term (see flow_eval_node's keyframe block —
        under the bias-consistent IMU model the anchor SOURCE, not just its
        weight, decides whether the gyro z-bias is observable)."""
        if not self.ok or not self.initialized:
            return None
        if self._pending_imu < 1:
            return (self.pose.translation(), self.vel) if self.pose is not None else None
        k = self.k + 1
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        graph.add(gtsam.CombinedImuFactor(X(k - 1), V(k - 1), X(k), V(k), B(k - 1), B(k), self.pim))
        nav = self.pim.predict(gtsam.NavState(self.pose, self.vel), self.bias)
        values.insert(X(k), nav.pose())
        values.insert(V(k), nav.velocity())
        values.insert(B(k), self.bias)
        if flow_vel_ned is not None:
            noise = (self.flow_noise if flow_sigma is None else
                     gtsam.noiseModel.Diagonal.Sigmas(np.array(
                         [max(float(flow_sigma), 1e-4),
                          max(float(flow_sigma), 1e-4), 1e6])))
            graph.add(gtsam.PriorFactorVector(V(k), np.asarray(flow_vel_ned, float), noise))
        graph.add(gtsam.GPSFactor(X(k), np.array([0.0, 0.0, float(depth)]), self.depth_noise))
        # Loose absolute attitude anchor from the IMU (see __init__). Without it the yaw
        # DOF is unobservable at low speed and the graph diverges. Position DOFs carry
        # huge sigma here, so this constrains ONLY orientation, never position.
        if self.att_prior and imu_quat_wxyz is not None:
            att_rot = gtsam.Rot3.Quaternion(*imu_quat_wxyz)
            att_pose = gtsam.Pose3(att_rot, nav.pose().translation())
            att_noise = (self.att_noise if att_yaw_sigma is None else
                         gtsam.noiseModel.Diagonal.Sigmas(np.array(
                             [self.att_rp_sigma, self.att_rp_sigma,
                              max(float(att_yaw_sigma), 1e-3),
                              1e6, 1e6, 1e6])))
            graph.add(gtsam.PriorFactorPose3(X(k), att_pose, att_noise))
        self.isam.update(graph, values)
        est = self.isam.calculateEstimate()
        self.pose = est.atPose3(X(k))
        self.vel = est.atVector(V(k))
        self.bias = est.atConstantBias(B(k))
        self.pim.resetIntegrationAndSetBias(self.bias)
        self._pending_imu = 0
        self.k = k
        # LANDMARK SLAM: a SECOND, isolated isam.update() with only landmark
        # factors, attached to the X(k) that now exists in the graph. Runs only when
        # the node buffered observations (landmark_mode='slam'); on any exception
        # the buffered obs are dropped and counted — the VIO chain above is already
        # committed and cannot be harmed.
        if self._lm_frame_obs:
            try:
                self._landmark_update(k)
            except Exception:
                self._lm_dropped += len(self._lm_frame_obs)
                self._lm_frame_obs = {}
        return self.pose.translation(), self.vel
    def current_ned_yaw(self):
        """The graph's OWN estimated NED yaw, or None before initialization.
        Verified against gtsam.Rot3: .yaw() extracts the Z-axis rotation directly
        (round-tripped through Rot3.Ypr in testing), consistent with the NED frame
        this graph runs in (MakeSharedD's n_gravity=(0,0,+g) convention).
        HISTORY NOTE (kept because the old claim was 'verified' and is now
        deliberately retired): the original IMU.cpp added yaw_drift only to the
        published orientation, leaving angular_velocity clean, so this attitude
        was drift-immune by construction. The LOCALLY MODIFIED IMU.cpp
        (2026-07-23) instead injects yaw_drift as a constant z-rate bias into the
        reported angular velocity and publishes its integral as yaw — so this
        attitude now drifts at the bias rate UNTIL the CombinedImuFactor's bias
        state converges, which requires an absolute yaw reference (the lane-
        anchored attitude prior). Rotating the flow measurement by THIS yaw is
        still correct — it is the graph's best current estimate and stays
        self-consistent as the bias converges — but 'immune' it no longer is."""
        if not self.ok or self.pose is None:
            return None
        return float(self.pose.rotation().yaw())
    def gyro_bias(self):
        """The graph's current estimate of the gyro bias [rad/s], (bx, by, bz),
        or None before initialization. HOW bz IS ACTUALLY DETERMINED (settled by
        run 230541 + the offline replica): the static-init MEASUREMENT at spawn
        is the mechanism — the bias prior is centred on it and the random walk
        holds it. Online refinement through iSAM2-with-default-relinearization
        is effectively stalled at this evidence scale (replica: bz moved 0.001
        deg/min in 300 s of lane-anchored keyframes against a true 1.00
        deg/min), and forcing relinearization is the known-divergent lever. So:
        if bz reads ~0, check the init lines — a skipped or thin static init
        (buffer raced by the first camera frame) cannot be recovered online.
        Cross-checks: |bz| vs the .scn yaw_drift and vs the EKF's b_psi (which
        DOES converge online, its lane updates being direct absolute yaw
        measurements); sign depends on the shim's body-axis convention, so
        compare magnitude first."""
        if not self.ok or not self.initialized:
            return None
        gb = self.bias.gyroscope()
        return (float(gb[0]), float(gb[1]), float(gb[2]))
    def add_landmark_xy(self, px, py, sigma_x, sigma_y):
        """Anisotropic absolute x/y correction from a mapped-landmark observation.
        Consumed by flow_eval_node's landmark_mode ('gate'/'map'); previously MISSING,
        so any run with landmark_mode != 'off' raised AttributeError. Adds a GPSFactor-
        style prior on the LATEST pose X(k) with per-axis sigma: pass a huge sigma on an
        axis to leave it untouched (gate x known, y randomized -> sigma_y=1e6). Depth is
        left free (z sigma huge) since pressure already constrains it. No-op until the
        graph is initialized and has at least one keyframe."""
        if not self.ok or not self.initialized or self.pose is None:
            return
        graph = gtsam.NonlinearFactorGraph()
        z = float(self.pose.translation()[2])
        noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([max(float(sigma_x), 1e-3), max(float(sigma_y), 1e-3), 1e6]))
        graph.add(gtsam.GPSFactor(X(self.k),
                                  np.array([float(px), float(py), z]), noise))
        self.isam.update(graph, gtsam.Values())
        est = self.isam.calculateEstimate()
        self.pose = est.atPose3(X(self.k))
        self.vel = est.atVector(V(self.k))
        self.bias = est.atConstantBias(B(self.k))
        return self.pose.translation(), self.vel
    # ------------------- LANDMARK SLAM (landmark_mode='slam') -------------------
    def set_landmark_prior(self, name, values_xyz, sigmas_xyz):
        """Rulebook knowledge, per axis, applied ONCE at the landmark's birth.
        values_xyz entries may be None (axis unknown -> the birth estimate is used
        as that axis' prior mean, with whatever sigma you pass, typically huge).
        Example — gate post, x known in the GRAPH frame (start-relative NED):
            set_landmark_prior('GatePostRed', (gx, None, None), (0.2, 1e3, 1e3))"""
        self._lm_priors[name] = (list(values_xyz), list(sigmas_xyz))
    def add_landmark_obs(self, name, rel_body_frd):
        """Buffer one body-FRD relative observation (from /vision/features). It is
        attached to the NEXT keyframe pose — matching the image that produced it up
        to one keyframe period (~0.2 s), same latency class as the flow prior.
        Returns 'ok' | 'range' | 'off' for the node's logging."""
        if not self.ok or not self.initialized:
            return 'off'
        rel = np.asarray(rel_body_frd, float)
        rng = float(np.linalg.norm(rel))
        if not (self.lm_min_range <= rng <= self.lm_max_range):
            return 'range'                       # GUARD: implausible range
        self._lm_frame_obs[name] = rel
        return 'ok'
    def _range_sigma(self, rng):
        a, b = self.lm_range_sigma
        return a + b * rng * rng
    def _lm_noise(self, rng):
        # BearingRangeFactor3D noise: [bearing(2), range(1)]. Huber so one bad
        # detection is down-weighted instead of dragging the map (GUARD).
        base = gtsam.noiseModel.Diagonal.Sigmas(np.array(
            [self.lm_bearing_sigma, self.lm_bearing_sigma, self._range_sigma(rng)]))
        return gtsam.noiseModel.Robust.Create(
            gtsam.noiseModel.mEstimator.Huber.Create(self.lm_huber_k), base)
    def _landmark_update(self, k):
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        for name, rel in self._lm_frame_obs.items():
            if name not in self._lm_ids:
                # GUARD(birth): only after lm_min_sightings buffered observations,
                # initialized at their MEDIAN composed with the current pose — the
                # first frame that ever sees a prop is usually the worst one, and
                # the map (and every later relocalization) inherits the first fix.
                hist = self._lm_hist.setdefault(name, [])
                hist.append(rel)
                if len(hist) < self.lm_min_sightings:
                    continue
                med = np.median(np.array(hist), axis=0)
                j = len(self._lm_ids)
                self._lm_ids[name] = j
                init_world = self.pose.transformFrom(gtsam.Point3(*med))
                values.insert(L(j), init_world)
                # birth prior: rulebook axes where given, else the birth estimate
                # with a weak stabilizing sigma (prevents a single-viewpoint
                # landmark making the system indeterminate before parallax).
                pv, ps = self._lm_priors.get(
                    name, ([None] * 3, [self.lm_default_prior_sigma] * 3))
                mean = np.array([init_world[i] if pv[i] is None else float(pv[i])
                                 for i in range(3)])
                graph.add(gtsam.PriorFactorPoint3(
                    L(j), gtsam.Point3(*mean),
                    gtsam.noiseModel.Diagonal.Sigmas(np.asarray(ps, float))))
            j = self._lm_ids[name]
            rng = float(np.linalg.norm(rel))
            graph.add(gtsam.BearingRangeFactor3D(
                X(k), L(j), gtsam.Unit3(gtsam.Point3(*rel)), rng,
                self._lm_noise(rng)))
        self._lm_frame_obs = {}
        if graph.size() == 0:
            return
        self.isam.update(graph, values)
        est = self.isam.calculateEstimate()
        self.pose = est.atPose3(X(k))
        self.vel = est.atVector(V(k))
        self.bias = est.atConstantBias(B(k))
    def landmark_estimates(self):
        """name -> np.array([x, y, z]) in the GRAPH frame (start-relative NED,
        origin at (0, 0, first depth)). Empty dict when no landmark has been born."""
        if not self.ok or not self._lm_ids:
            return {}
        est = self.isam.calculateEstimate()
        out = {}
        for name, j in self._lm_ids.items():
            if est.exists(L(j)):
                out[name] = np.asarray(est.atPoint3(L(j)))
        return out
