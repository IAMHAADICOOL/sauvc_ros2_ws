#!/usr/bin/env python3
"""imu_taobotics_check — Pre-Phase bring-up check for the Taobotics HFI-A9.

Prerequisite: the mrpt Taobotics driver must already be running and REMAPPED to
publish on /imu/data (its default publish_topic is literally "sensor", not /imu/data —
see SETUP.md Pre-Phase for the full launch command). This script only subscribes and
sanity-checks; it doesn't talk to the serial port itself.

Usage:
  ros2 run sauvc_sensor_check imu_taobotics_check --ros-args -p topic:=/imu/data
"""
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu

from sauvc_sensor_check.imu_check_common import quat_to_euler_deg, accel_sanity


class ImuTaoboticsCheckNode(Node):
    def __init__(self):
        super().__init__('imu_taobotics_check')
        self.declare_parameter('topic', '/imu/data')
        topic = self.get_parameter('topic').value
        self.n = 0
        self.t0 = None
        self.last_print = 0.0
        self.create_subscription(Imu, topic, self.on_imu, 20)
        self.get_logger().info(f'listening on {topic} ... (waiting for first message; '
                               f'if nothing arrives in ~10s, is the mrpt driver launched '
                               f'and remapped to this topic? `ros2 topic list` to check)')
        self.create_timer(10.0, self.check_silence)

    def check_silence(self):
        if self.n == 0:
            self.get_logger().warn(
                'still no messages. Checklist: (1) is '
                'mrpt_sensor_imu_taobotics.launch.py running with publish_topic:='
                '<this topic>? (2) `ros2 topic list` — does the topic exist at all? '
                '(3) is serial_port correct (`ls /dev/ttyUSB*`)? (4) A9 takes ~10s to '
                'initialize its magnetic-field baseline after power-on.')

    def on_imu(self, msg):
        if self.t0 is None:
            self.t0 = time.time()
        self.n += 1
        q = msg.orientation
        roll, pitch, yaw, norm_ok = quat_to_euler_deg(q.x, q.y, q.z, q.w)
        a = msg.linear_acceleration
        amag, accel_ok = accel_sanity(a.x, a.y, a.z)

        now = time.time()
        if now - self.last_print < 0.5:   # print at most 2 Hz even if data is faster
            return
        self.last_print = now
        elapsed = now - self.t0
        hz = self.n / elapsed if elapsed > 0 else 0.0

        flags = []
        if not norm_ok:
            flags.append('QUATERNION NOT NORMALIZED — check driver/message parsing')
        if accel_ok is False:
            flags.append(f'accel magnitude {amag:.2f} m/s^2 far from ~9.8 — check mounting/units')
        elif accel_ok is None:
            flags.append('linear_acceleration looks unpopulated (AHRS-only output?)')
        flag_str = ('  <-- ' + '; '.join(flags)) if flags else ''

        self.get_logger().info(
            f'rpy=({roll:+6.1f},{pitch:+6.1f},{yaw:+6.1f}) deg   '
            f'gyro=({msg.angular_velocity.x:+.3f},{msg.angular_velocity.y:+.3f},'
            f'{msg.angular_velocity.z:+.3f}) rad/s   {hz:.1f} Hz (n={self.n}){flag_str}')


def main():
    rclpy.init()
    node = ImuTaoboticsCheckNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
