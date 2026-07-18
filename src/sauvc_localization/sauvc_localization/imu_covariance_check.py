#!/usr/bin/env python3
"""imu_covariance_check.py — Phase 2b. Two jobs:

1. DIAGNOSE: print the covariance fields actually arriving on /imu/data. If they're all
   zero (common with imu_filter_madgwick unless configured) or -1 (ROS convention for
   "unknown"), the EKF is either treating the IMU as perfectly noise-free (bad — it will
   dominate every other sensor and the filter will barely fuse anything else) or falling
   back to a hardcoded internal default that has nothing to do with YOUR hardware.

2. PATCH (optional): if the upstream driver can't be configured to publish real
   covariance, this node re-publishes /imu/data with covariance fields overwritten from
   parameters you set using the values from estimate_covariance.py. Point sauvc_bringup's
   phase2 launch at /imu/data_corrected instead of /imu/data if you use this.

madgwick note: imu_filter_madgwick DOES expose real covariance if you set its
`orientation_stddev` parameter — try that route first (one parameter, no extra node)
before reaching for this patch node.
"""
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu


class ImuCovarianceCheck(Node):
    def __init__(self):
        super().__init__('imu_covariance_check')
        self.declare_parameter('patch', False)
        self.declare_parameter('orientation_var_rpy', [0.001, 0.001, 0.0002])  # rad^2
        self.declare_parameter('gyro_var_xyz', [0.0001, 0.0001, 0.0001])       # (rad/s)^2
        self.declare_parameter('accel_var_xyz', [0.05, 0.05, 0.05])           # (m/s^2)^2
        self.patch = self.get_parameter('patch').value
        self.printed_diagnosis = False
        self.n = 0
        self.pub = self.create_publisher(Imu, '/imu/data_corrected', 10) if self.patch else None
        self.create_subscription(Imu, '/imu/data', self.on_imu, 50)

    def on_imu(self, msg):
        self.n += 1
        if not self.printed_diagnosis and self.n > 20:
            self.printed_diagnosis = True
            self._diagnose(msg)
        if self.patch:
            self._republish(msg)

    def _diagnose(self, msg):
        oc = list(msg.orientation_covariance)
        gc = list(msg.angular_velocity_covariance)
        ac = list(msg.linear_acceleration_covariance)
        for name, cov in [('orientation', oc), ('angular_velocity', gc),
                          ('linear_acceleration', ac)]:
            diag = [cov[0], cov[4], cov[8]]
            if all(v == 0.0 for v in cov):
                verdict = 'ALL ZERO -> EKF will treat this as perfect/no-noise. FIX THIS.'
            elif diag[0] < 0:
                verdict = 'UNKNOWN (-1 convention) -> EKF falls back to internal default. FIX THIS.'
            else:
                verdict = 'looks populated, sanity-check the numbers below against your own measurement'
            self.get_logger().info(f'{name} diag covariance = {diag}  -> {verdict}')
        if not self.patch:
            self.get_logger().info(
                'If any said FIX THIS: try imu_filter_madgwick "orientation_stddev" '
                'param first, or set patch:=true on this node with your measured '
                'variances from estimate_covariance.py.')

    def _republish(self, msg):
        out = msg
        ov = self.get_parameter('orientation_var_rpy').value
        gv = self.get_parameter('gyro_var_xyz').value
        av = self.get_parameter('accel_var_xyz').value
        oc = [0.0]*9; oc[0], oc[4], oc[8] = ov
        gc = [0.0]*9; gc[0], gc[4], gc[8] = gv
        ac = [0.0]*9; ac[0], ac[4], ac[8] = av
        out.orientation_covariance = oc
        out.angular_velocity_covariance = gc
        out.linear_acceleration_covariance = ac
        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(ImuCovarianceCheck())


if __name__ == '__main__':
    main()
