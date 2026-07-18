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
                 flow_sigma=0.05, depth_sigma=0.002, compare_frame='ned'):
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
        self.flow_noise = gtsam.noiseModel.Isotropic.Sigma(3, flow_sigma)
        self.depth_noise = gtsam.noiseModel.Diagonal.Sigmas(np.array([1e6, 1e6, depth_sigma]))
        self.isam = gtsam.ISAM2(gtsam.ISAM2Params())
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

    def add_keyframe(self, flow_vel_ned, depth):
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
        self.isam.update(graph, values)
        est = self.isam.calculateEstimate()
        self.pose = est.atPose3(X(k))
        self.vel = est.atVector(V(k))
        self.bias = est.atConstantBias(B(k))
        self.pim.resetIntegrationAndSetBias(self.bias)
        self._pending_imu = 0
        self.k = k
        return self.pose.translation(), self.vel
