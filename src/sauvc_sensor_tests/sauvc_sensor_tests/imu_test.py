#!/usr/bin/env python3
"""IMU test: prints vehicle orientation (roll/pitch/yaw, degrees) plus raw
angular velocity [rad/s] and linear acceleration [m/s^2] in the terminal.

Run:  ros2 run sauvc_sensor_tests imu_test
      ros2 run sauvc_sensor_tests imu_test --ros-args -p topic:=/sauvc_auv/imu
"""
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


def quat_to_rpy(x, y, z, w):
    # ZYX convention (yaw about Z, NED)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    s = max(-1.0, min(1.0, 2 * (w * y - z * x)))
    pitch = math.asin(s)
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


class ImuTest(Node):
    def __init__(self):
        super().__init__('imu_test')
        self.declare_parameter('topic', '/sauvc_auv/imu')
        topic = self.get_parameter('topic').value
        self.sub = self.create_subscription(Imu, topic, self.cb, qos_profile_sensor_data)
        self.get_logger().info(f'listening on {topic}')

    def cb(self, m: Imu):
        q = m.orientation
        r, p, y = (math.degrees(a) for a in quat_to_rpy(q.x, q.y, q.z, q.w))
        w, a = m.angular_velocity, m.linear_acceleration
        print(f'\rRPY [deg]: {r:+7.2f} {p:+7.2f} {y:+7.2f} | '
              f'gyro [rad/s]: {w.x:+6.3f} {w.y:+6.3f} {w.z:+6.3f} | '
              f'accel [m/s2]: {a.x:+7.3f} {a.y:+7.3f} {a.z:+7.3f}',
              end='', flush=True)


def main():
    rclpy.init()
    rclpy.spin(ImuTest())


if __name__ == '__main__':
    main()
