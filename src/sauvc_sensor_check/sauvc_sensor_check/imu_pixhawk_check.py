#!/usr/bin/env python3
"""imu_pixhawk_check — Pre-Phase bring-up check for the Pixhawk IMU via mavros.

Prerequisite: mavros must already be running and connected to the Pixhawk
(`ros2 launch mavros apm.launch.py fcu_url:=/dev/pixhawk:57600` once pinned via SETUP.md
section 4, or `fcu_url:=/dev/ttyACM0:57600` for a first test before pinning). This script subscribes to /mavros/imu/data (fused orientation) AND
/mavros/state (connection/arm status) — the latter catches "mavros is running but never
actually talked to the FCU" before you waste time debugging the IMU topic.

Usage:
  ros2 run sauvc_sensor_check imu_pixhawk_check --ros-args -p topic:=/mavros/imu/data
"""
import time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from mavros_msgs.msg import State

from sauvc_sensor_check.imu_check_common import quat_to_euler_deg, accel_sanity


class ImuPixhawkCheckNode(Node):
    def __init__(self):
        super().__init__('imu_pixhawk_check')
        self.declare_parameter('topic', '/mavros/imu/data')
        topic = self.get_parameter('topic').value
        self.n = 0
        self.t0 = None
        self.last_print = 0.0
        self.fcu_connected = None   # None = no /mavros/state message seen yet

        self.create_subscription(State, '/mavros/state', self.on_state, 10)
        self.create_subscription(Imu, topic, self.on_imu, 20)
        self.get_logger().info(f'listening on {topic} and /mavros/state ...')
        self.create_timer(10.0, self.check_silence)

    def check_silence(self):
        if self.fcu_connected is None:
            self.get_logger().warn(
                'no /mavros/state seen — is mavros running at all? '
                '`ros2 node list | grep mavros`')
        elif self.fcu_connected is False:
            self.get_logger().warn(
                'mavros is running but connected=false — FCU not talking to mavros. '
                'Checklist: (1) fcu_url matches the actual device (`ls /dev/ttyACM*` or '
                '/dev/ttyUSB*), (2) Pixhawk is powered, (3) try unplugging/replugging '
                'USB, (4) baud rate — USB (ACM) usually ignores baud, try omitting it.')
        if self.n == 0 and self.fcu_connected:
            self.get_logger().warn(
                f'mavros IS connected to the FCU but no messages on {topic} — '
                f'check ArduSub has IMU streaming enabled at a nonzero rate '
                f'(SR params) and that this is the right topic (`ros2 topic list`).')

    def on_state(self, msg):
        self.fcu_connected = msg.connected
        if not msg.connected:
            return
        # Only log mode/armed occasionally via the IMU print path below to avoid spam;
        # store latest for that print.
        self.mode = msg.mode
        self.armed = msg.armed

    def on_imu(self, msg):
        if self.t0 is None:
            self.t0 = time.time()
        self.n += 1
        q = msg.orientation
        roll, pitch, yaw, norm_ok = quat_to_euler_deg(q.x, q.y, q.z, q.w)
        a = msg.linear_acceleration
        amag, accel_ok = accel_sanity(a.x, a.y, a.z)

        now = time.time()
        if now - self.last_print < 0.5:
            return
        self.last_print = now
        elapsed = now - self.t0
        hz = self.n / elapsed if elapsed > 0 else 0.0

        flags = []
        if not norm_ok:
            flags.append('QUATERNION NOT NORMALIZED')
        if accel_ok is False:
            flags.append(f'accel magnitude {amag:.2f} m/s^2 far from ~9.8')
        conn_str = getattr(self, 'mode', '?') if self.fcu_connected else 'DISCONNECTED'
        flag_str = ('  <-- ' + '; '.join(flags)) if flags else ''

        self.get_logger().info(
            f'rpy=({roll:+6.1f},{pitch:+6.1f},{yaw:+6.1f}) deg   '
            f'gyro=({msg.angular_velocity.x:+.3f},{msg.angular_velocity.y:+.3f},'
            f'{msg.angular_velocity.z:+.3f}) rad/s   mode={conn_str}   '
            f'{hz:.1f} Hz (n={self.n}){flag_str}')


def main():
    rclpy.init()
    node = ImuPixhawkCheckNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
