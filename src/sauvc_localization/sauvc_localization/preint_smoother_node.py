#!/usr/bin/env python3
"""preint_smoother_node — Phase 7 (optional). Factor-graph smoother with IMU
preintegration (GTSAM), fusing preintegrated IMU + optical-flow velocity + depth.

Why: between camera keyframes, all high-rate IMU samples are compressed into one
relative-motion factor with bias Jacobians (Forster et al. preintegration). The smoother
estimates accel/gyro biases online while flow is healthy, so during FLOW DROPOUTS
(caustics, featureless floor, pitching at a flare) it coasts on bias-corrected IMU with
bounded error instead of coasting blind like the EKF.

Run it IN PARALLEL with the robot_localization EKF and A/B them on the same bags:
  /odometry/filtered  (EKF, Phase 5)   vs   /odometry/preint  (this node)

Graph per keyframe (default 5 Hz):
  ImuFactor(X_i-1, V_i-1, X_i, V_i, B_i-1)      <- preintegrated IMU
  BetweenFactor(B_i-1, B_i, 0)                   <- bias random walk
  PriorFactorVector(V_i, R*[vx,vy,0])            <- flow velocity (when fresh)
  GPSFactor(X_i, [x_est, y_est, -depth])         <- depth only (x,y sigmas huge)
Solved incrementally with ISAM2; window reset every reset_s seconds bounds compute.

Install:  pip3 install gtsam==4.3a0     (has Jetson/aarch64 + Python 3.10 wheels)
"""
import math
import numpy as np

try:
    import gtsam
    from gtsam.symbol_shorthand import X, V, B
    GTSAM_OK = True
except ImportError:
    GTSAM_OK = False


class PreintFusionCore:
    """Pure-python core (no ROS). Feed IMU/flow/depth with timestamps; read .state."""

    def __init__(self, accel_sigma=0.15, gyro_sigma=0.01,
                 accel_bias_rw=1e-3, gyro_bias_rw=1e-4,
                 keyframe_dt=0.2, reset_s=60.0, gravity=9.81):
        if not GTSAM_OK:
            raise RuntimeError('pip3 install gtsam==4.3a0')
        self.keyframe_dt = keyframe_dt
        self.reset_s = reset_s

        p = gtsam.PreintegrationParams.MakeSharedU(gravity)
        p.setAccelerometerCovariance(np.eye(3) * accel_sigma**2)
        p.setGyroscopeCovariance(np.eye(3) * gyro_sigma**2)
        p.setIntegrationCovariance(np.eye(3) * 1e-8)
        self.pim_params = p
        self.bias_rw_noise = gtsam.noiseModel.Diagonal.Sigmas(
            np.array([accel_bias_rw]*3 + [gyro_bias_rw]*3))

        self.latest_flow = None    # (t, vx, vy, var)
        self.latest_depth = None   # (t, z, var)
        self.last_imu_t = None
        self.kf_t = None
        self.i = 0
        self.state = None          # dict(t, p, R, v, bias) — the output
        self._start_graph(gtsam.Pose3(), np.zeros(3), gtsam.imuBias.ConstantBias())

    def _start_graph(self, pose, vel, bias):
        self.isam = gtsam.ISAM2()
        graph = gtsam.NonlinearFactorGraph()
        vals = gtsam.Values()
        self.i = 0
        graph.add(gtsam.PriorFactorPose3(X(0), pose,
                  gtsam.noiseModel.Diagonal.Sigmas(
                      np.array([0.05, 0.05, 0.05, 0.5, 0.5, 0.05]))))  # rpy, xyz
        graph.add(gtsam.PriorFactorVector(V(0), vel,
                  gtsam.noiseModel.Isotropic.Sigma(3, 0.05)))
        graph.add(gtsam.PriorFactorConstantBias(B(0), bias,
                  gtsam.noiseModel.Isotropic.Sigma(6, 0.1)))
        vals.insert(X(0), pose)
        vals.insert(V(0), vel)
        vals.insert(B(0), bias)
        self.isam.update(graph, vals)
        self.prev_state = gtsam.NavState(pose, vel)
        self.prev_bias = bias
        self.pim = gtsam.PreintegratedImuMeasurements(self.pim_params, bias)
        self.graph_t0 = None

    # ------------------------------------------------------------------ inputs
    def add_flow(self, t, vx, vy, var):
        self.latest_flow = (t, vx, vy, max(var, 1e-4))

    def add_depth(self, t, z, var):
        self.latest_depth = (t, z, max(var, 1e-6))

    def add_imu(self, t, acc, gyr):
        """acc: specific force [m/s^2] body frame (includes gravity reaction),
           gyr: angular rate [rad/s] body frame. Call at full IMU rate."""
        if self.last_imu_t is None:
            self.last_imu_t = t
            self.kf_t = t
            self.graph_t0 = t
            return
        dt = t - self.last_imu_t
        self.last_imu_t = t
        if dt <= 0 or dt > 0.1:
            return
        self.pim.integrateMeasurement(np.asarray(acc, float),
                                      np.asarray(gyr, float), dt)
        if t - self.kf_t >= self.keyframe_dt:
            self._keyframe(t)

    # ------------------------------------------------------------------ keyframe
    def _keyframe(self, t):
        j = self.i + 1
        graph = gtsam.NonlinearFactorGraph()
        vals = gtsam.Values()

        graph.add(gtsam.ImuFactor(X(self.i), V(self.i), X(j), V(j), B(self.i),
                                  self.pim))
        graph.add(gtsam.BetweenFactorConstantBias(
            B(self.i), B(j), gtsam.imuBias.ConstantBias(), self.bias_rw_noise))

        pred = self.pim.predict(self.prev_state, self.prev_bias)
        vals.insert(X(j), pred.pose())
        vals.insert(V(j), pred.velocity())
        vals.insert(B(j), self.prev_bias)

        # Flow velocity factor (body -> world with current attitude), if fresh (<0.3 s)
        if self.latest_flow and t - self.latest_flow[0] < 0.3:
            _, vx, vy, var = self.latest_flow
            Rwb = pred.pose().rotation().matrix()
            vw = Rwb @ np.array([vx, vy, 0.0])
            sig = math.sqrt(var)
            graph.add(gtsam.PriorFactorVector(V(j), vw,
                      gtsam.noiseModel.Diagonal.Sigmas(np.array([sig, sig, 1.0]))))

        # Depth factor: z tight, x/y sigmas huge => constrains ONLY z (same
        # anisotropic-covariance idea as the gate x-correction).
        if self.latest_depth and t - self.latest_depth[0] < 0.3:
            _, z, var = self.latest_depth
            pos = pred.pose().translation()
            graph.add(gtsam.GPSFactor(X(j),
                      gtsam.Point3(pos[0], pos[1], z),
                      gtsam.noiseModel.Diagonal.Sigmas(
                          np.array([50.0, 50.0, math.sqrt(var)]))))

        self.isam.update(graph, vals)
        est = self.isam.calculateEstimate()
        pose = est.atPose3(X(j))
        vel = est.atVector(V(j))
        bias = est.atConstantBias(B(j))

        self.prev_state = gtsam.NavState(pose, vel)
        self.prev_bias = bias
        self.pim = gtsam.PreintegratedImuMeasurements(self.pim_params, bias)
        self.i = j
        self.kf_t = t
        self.state = dict(t=t, p=np.asarray(pose.translation()),
                          R=pose.rotation().matrix(), v=np.asarray(vel), bias=bias)

        # Sliding-window reset: bound ISAM2 memory/latency over a 15-min mission.
        if t - self.graph_t0 > self.reset_s:
            self._start_graph(pose, vel, bias)
            self.graph_t0 = t


# ======================= ROS 2 wrapper =======================
def main():
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Imu
    from geometry_msgs.msg import TwistWithCovarianceStamped, PoseWithCovarianceStamped
    from nav_msgs.msg import Odometry

    class PreintSmootherNode(Node):
        def __init__(self):
            super().__init__('preint_smoother_node')
            self.declare_parameter('accel_sigma', 0.15)   # m/s^2, INFLATE if thrusters
            self.declare_parameter('gyro_sigma', 0.01)    # rad/s     shake the IMU
            self.declare_parameter('keyframe_dt', 0.2)
            self.declare_parameter('reset_s', 60.0)
            g = lambda n: self.get_parameter(n).value
            self.core = PreintFusionCore(accel_sigma=g('accel_sigma'),
                                         gyro_sigma=g('gyro_sigma'),
                                         keyframe_dt=g('keyframe_dt'),
                                         reset_s=g('reset_s'))
            self.pub = self.create_publisher(Odometry, '/odometry/preint', 10)
            self.create_subscription(Imu, '/imu/data', self.on_imu, 100)
            self.create_subscription(TwistWithCovarianceStamped, '/flow/twist',
                                     self.on_flow, 10)
            self.create_subscription(PoseWithCovarianceStamped, '/depth',
                                     self.on_depth, 10)
            self.get_logger().info('preint_smoother_node up (gtsam)')

        @staticmethod
        def _t(msg):
            return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        def on_flow(self, msg):
            self.core.add_flow(self._t(msg), msg.twist.twist.linear.x,
                               msg.twist.twist.linear.y, msg.twist.covariance[0])

        def on_depth(self, msg):
            self.core.add_depth(self._t(msg), msg.pose.pose.position.z,
                                msg.pose.covariance[14])

        def on_imu(self, msg):
            acc = (msg.linear_acceleration.x, msg.linear_acceleration.y,
                   msg.linear_acceleration.z)
            gyr = (msg.angular_velocity.x, msg.angular_velocity.y,
                   msg.angular_velocity.z)
            self.core.add_imu(self._t(msg), acc, gyr)
            s = self.core.state
            if s is None or s['t'] != self.core.kf_t:
                return
            o = Odometry()
            o.header.stamp = msg.header.stamp
            o.header.frame_id = 'odom'
            o.child_frame_id = 'base_link'
            o.pose.pose.position.x, o.pose.pose.position.y, o.pose.pose.position.z = s['p']
            # rotation matrix -> quaternion
            Rm = s['R']
            qw = math.sqrt(max(1 + Rm[0, 0] + Rm[1, 1] + Rm[2, 2], 1e-9)) / 2
            o.pose.pose.orientation.w = qw
            o.pose.pose.orientation.x = (Rm[2, 1] - Rm[1, 2]) / (4 * qw)
            o.pose.pose.orientation.y = (Rm[0, 2] - Rm[2, 0]) / (4 * qw)
            o.pose.pose.orientation.z = (Rm[1, 0] - Rm[0, 1]) / (4 * qw)
            o.twist.twist.linear.x, o.twist.twist.linear.y, o.twist.twist.linear.z = s['v']
            self.pub.publish(o)

    rclpy.init()
    rclpy.spin(PreintSmootherNode())


if __name__ == '__main__':
    main()
