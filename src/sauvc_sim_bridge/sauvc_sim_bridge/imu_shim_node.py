#!/usr/bin/env python3
"""imu_shim_node — makes Stonefish's IMU look like the Taobotics HFI-A9.

Sub: /sauvc_auv/imu   sensor_msgs/Imu   (NED world, FRD body)  -- simulator's
Pub: /imu/data        sensor_msgs/Imu   (ENU world, FLU body)  -- what your stack expects

This is a shim, not a driver. `sauvc_localization` and `robot_localization` must not be
able to tell whether this or the real HFI-A9 is upstream. Nothing here is allowed to
know anything the real driver could not know -- in particular it never touches
/sauvc_auv/odometry.

Verified facts about the upstream publisher (read from
stonefish_ros2/src/stonefish_ros2/ROS2Interface.cpp, not from the docs):

  * PublishIMU DOES fill `orientation` -- sample channels 0..2 are roll/pitch/yaw and
    are assembled into a quaternion. Stonefish's IMU is an AHRS, so it is a genuine
    analogue of the HFI-A9 and the Madgwick filter stays skipped, exactly as on
    hardware. (The sim README's "specific force + rates" line is wrong.)
  * `orientation_covariance` is sigma^2 of the scene's <noise angle=...>, likewise
    `angular_velocity_covariance` from <noise angular_velocity=...>. Confirmed against
    a live echo: 3.045025e-12 == 1.745e-6^2.
  * `linear_acceleration_covariance` is ALL ZEROS whenever the scene omits a
    <noise linear_acceleration=...> attribute -- which yours does. Per the
    sensor_msgs/Imu spec, zero means "perfectly known", not "unknown" (unknown is -1).
    Harmless today because ekf.yaml does not fuse acceleration, but it is a loaded gun:
    a zero variance is a singular measurement covariance. This node rewrites it to -1
    so that anything downstream that starts fusing accel fails loudly instead of
    silently inverting a singular matrix.
  * `header.stamp` is `nh_->get_clock()->now()` -- the WALL CLOCK at publish time. The
    sample's own timestamp is discarded (0 uses of s.getTimestamp() in the whole ROS2
    file, vs 20 uses of get_clock()->now()). There is no /clock. So use_sim_time must be
    FALSE everywhere, and it also means sensor stamps carry publish jitter rather than
    sample time. See rtf_monitor_node for why this matters more than it looks.

Frame conversion is delegated entirely to frames.py. No sign flip is written by hand
here; if the attitude is wrong, fix frames.py, not this file.
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

from sauvc_sim_bridge.frames import (
    ned_frd_quat_to_enu_flu, frd_to_flu_vec, cov3_frd_to_flu, cov3_ned_to_enu,
)


class ImuShimNode(Node):
    def __init__(self):
        super().__init__('imu_shim_node')
        self.declare_parameter('in_topic', '/sauvc_auv/imu')
        self.declare_parameter('out_topic', '/imu/data')
        self.declare_parameter('frame_id', 'imu_link')
        # The real HFI-A9's fused yaw drifts; Stonefish's does not unless the scene sets
        # <noise yaw_drift=...>. If you have not set it, this node refuses to pretend
        # the problem is solved -- see the warning below.
        self.declare_parameter('warn_if_no_drift', True)
        self.declare_parameter('drift_warn_cov_threshold', 1e-8)  # rad^2

        g = lambda n: self.get_parameter(n).value
        self.frame_id = g('frame_id')
        self.warned = False
        self.warn_enabled = g('warn_if_no_drift')
        self.cov_thresh = g('drift_warn_cov_threshold')

        self.pub = self.create_publisher(Imu, g('out_topic'), 50)
        self.create_subscription(Imu, g('in_topic'), self.on_imu, 50)
        self.get_logger().info(
            f"imu_shim: {g('in_topic')} (NED/FRD) -> {g('out_topic')} (ENU/FLU)")

    def on_imu(self, msg):
        out = Imu()
        # Pass the stamp through untouched. It is already wall clock, and the ArduSub
        # bridge regression came from synthesising timestamps -- do not repeat that.
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id

        q = ned_frd_quat_to_enu_flu(
            [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        out.orientation.x, out.orientation.y, out.orientation.z, out.orientation.w = q

        # Orientation covariance is expressed about the WORLD axes -> NED->ENU.
        out.orientation_covariance = list(cov3_ned_to_enu(list(msg.orientation_covariance)))

        w = frd_to_flu_vec([msg.angular_velocity.x,
                            msg.angular_velocity.y,
                            msg.angular_velocity.z])
        out.angular_velocity.x, out.angular_velocity.y, out.angular_velocity.z = w
        # Angular velocity is a BODY quantity -> FRD->FLU.
        out.angular_velocity_covariance = list(
            cov3_frd_to_flu(list(msg.angular_velocity_covariance)))

        a = frd_to_flu_vec([msg.linear_acceleration.x,
                            msg.linear_acceleration.y,
                            msg.linear_acceleration.z])
        out.linear_acceleration.x, out.linear_acceleration.y, out.linear_acceleration.z = a

        acc_cov = list(cov3_frd_to_flu(list(msg.linear_acceleration_covariance)))
        if all(abs(c) < 1e-30 for c in acc_cov):
            # Translate Stonefish's "no noise configured" (all zeros) into the ROS
            # spec's "no estimate" sentinel, rather than claiming infinite certainty.
            acc_cov = [-1.0] + [0.0] * 8
        out.linear_acceleration_covariance = acc_cov

        self._maybe_warn_no_drift(msg)
        self.pub.publish(out)

    def _maybe_warn_no_drift(self, msg):
        """Say out loud, once, if the sim IMU is too good to test lane_heading_node.

        Stonefish's orientation is ground truth + white noise on the angle. It has no
        bias walk. The ONLY mechanism that makes sim yaw drift is the scene's
        <noise yaw_drift="..."> attribute, and yaw_drift does not show up in any
        published field -- so this heuristic keys off the angle variance instead: a
        variance this small means the scene is running a near-perfect AHRS, in which
        case lane_heading_node has nothing to correct and every test of it passes
        vacuously.
        """
        if self.warned or not self.warn_enabled:
            return
        yaw_var = msg.orientation_covariance[8]
        if yaw_var < self.cov_thresh:
            self.warned = True
            self.get_logger().warn(
                f'sim IMU yaw variance is {yaw_var:.3e} rad^2 (sigma={yaw_var**0.5:.2e} rad). '
                'This is a near-perfect AHRS. Stonefish orientation is ground truth + '
                'white noise with NO bias walk, so unless my_auv.scn sets '
                '<noise ... yaw_drift="R"/>, sim yaw CANNOT drift and lane_heading_node '
                'has nothing to correct -- its tests will pass vacuously. Put your '
                'measured Phase 2 drift (deg/min -> rad/s) into the scene.')


def main():
    rclpy.init()
    rclpy.spin(ImuShimNode())


if __name__ == '__main__':
    main()
