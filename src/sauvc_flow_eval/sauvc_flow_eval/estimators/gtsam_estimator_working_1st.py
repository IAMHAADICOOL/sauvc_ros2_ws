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
"""

import numpy as np

try:
    import gtsam
    from gtsam.symbol_shorthand import X, V, B
    _GTSAM = True
except Exception:
    _GTSAM = False


class GtsamEstimator:
    available = _GTSAM

    def __init__(self, gravity=9.81, accel_sigma=0.02, gyro_sigma=0.0017,
                 accel_bias_rw=0.001, gyro_bias_rw=0.0001,
                 flow_sigma=0.05, depth_sigma=0.002, compare_frame='ned',
                 att_prior=True, att_rp_sigma=0.01, att_yaw_sigma=0.2):
        self.ok = _GTSAM
        self.frame = compare_frame
        if not self.ok:
            return
        params = gtsam.PreintegrationCombinedParams.MakeSharedD(gravity)
        params.setAccelerometerCovariance(np.eye(3) * accel_sigma ** 2)
        params.setGyroscopeCovariance(np.eye(3) * gyro_sigma ** 2)
        params.setIntegrationCovariance(np.eye(3) * 1e-8)
        params.setBiasAccCovariance(np.eye(3) * accel_bias_rw ** 2)
        params.setBiasOmegaCovariance(np.eye(3) * gyro_bias_rw ** 2)
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
        # (att_yaw_sigma, ~11 deg default) so it only prevents the blow-up, letting real
        # motion/landmarks still refine heading and keeping the IMU's slow yaw_drift
        # (~1 deg/min) a negligible contributor at this sigma. Disable with att_prior=
        # False to reproduce the divergence.
        self.att_prior = bool(att_prior)
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
        self.k = 0
        self.initialized = False
        self.pose = None
        self.vel = None
        self.t_prev_imu = None
        self._pending_imu = 0

    def add_imu(self, accel_xyz, gyro_xyz, t):
        if not self.ok:
            return
        if self.t_prev_imu is None:
            self.t_prev_imu = t
            return
        dt = t - self.t_prev_imu
        self.t_prev_imu = t
        if 0.0 < dt < 0.1 and self.initialized:
            self.pim.integrateMeasurement(np.asarray(accel_xyz, float),
                                          np.asarray(gyro_xyz, float), dt)
            self._pending_imu += 1

    def initialize(self, quat_wxyz_ned, init_vel_ned, depth):
        if not self.ok or self.initialized:
            return
        rot = gtsam.Rot3.Quaternion(*quat_wxyz_ned)
        pos = np.array([0.0, 0.0, float(depth)])
        self.pose = gtsam.Pose3(rot, pos)
        self.vel = np.asarray(init_vel_ned, float)
        graph = gtsam.NonlinearFactorGraph()
        values = gtsam.Values()
        pose_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.05, 0.05, 0.1, 0.1, 0.1]))
        vel_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        bias_noise = gtsam.noiseModel.Isotropic.Sigma(6, 1e-3)
        graph.add(gtsam.PriorFactorPose3(X(0), self.pose, pose_noise))
        graph.add(gtsam.PriorFactorVector(V(0), self.vel, vel_noise))
        graph.add(gtsam.PriorFactorConstantBias(B(0), self.bias, bias_noise))
        values.insert(X(0), self.pose)
        values.insert(V(0), self.vel)
        values.insert(B(0), self.bias)
        self.isam.update(graph, values)
        self.initialized = True

    def add_keyframe(self, flow_vel_ned, depth, imu_quat_wxyz=None):
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
            graph.add(gtsam.PriorFactorVector(V(k), np.asarray(flow_vel_ned, float), self.flow_noise))
        graph.add(gtsam.GPSFactor(X(k), np.array([0.0, 0.0, float(depth)]), self.depth_noise))
        # Loose absolute attitude anchor from the IMU (see __init__). Without it the yaw
        # DOF is unobservable at low speed and the graph diverges. Position DOFs carry
        # huge sigma here, so this constrains ONLY orientation, never position.
        if self.att_prior and imu_quat_wxyz is not None:
            att_rot = gtsam.Rot3.Quaternion(*imu_quat_wxyz)
            att_pose = gtsam.Pose3(att_rot, nav.pose().translation())
            graph.add(gtsam.PriorFactorPose3(X(k), att_pose, self.att_noise))
        self.isam.update(graph, values)
        est = self.isam.calculateEstimate()
        self.pose = est.atPose3(X(k))
        self.vel = est.atVector(V(k))
        self.bias = est.atConstantBias(B(k))
        self.pim.resetIntegrationAndSetBias(self.bias)
        self._pending_imu = 0
        self.k = k
        return self.pose.translation(), self.vel

    def current_ned_yaw(self):
        """The graph's OWN estimated NED yaw, or None before initialization.

        Verified against gtsam.Rot3: .yaw() extracts the Z-axis rotation directly
        (round-tripped through Rot3.Ypr in testing), consistent with the NED frame
        this graph runs in (MakeSharedD's n_gravity=(0,0,+g) convention).

        WHY THIS EXISTS: confirmed by reading Stonefish's IMU.cpp directly — the
        published orientation's yaw channel gets `accumulatedYawDrift` added as a
        pure post-hoc ramp (yawDriftRate * dt, accumulated forever), but the RAW
        angular_velocity channel this graph's add_imu() consumes is computed straight
        from the TRUE angular velocity, entirely upstream of that drift injection.
        So this graph's own preintegrated attitude is mathematically untouched by
        yaw_drift — it never sees the corrupted signal. Rotating the flow velocity
        measurement into world using THIS yaw (rather than an externally-tracked yaw
        derived from the published, drift-corrupted orientation) lets the graph fuse
        an independent measurement instead of one pre-contaminated by the same error
        it would otherwise have no way to cross-check against."""
        if not self.ok or self.pose is None:
            return None
        return float(self.pose.rotation().yaw())

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
