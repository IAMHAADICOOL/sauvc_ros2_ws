#!/usr/bin/env python3
"""
ArduSub SITL <-> Stonefish bridge (ArduPilot "JSON" SITL backend).

ArduPilot SITL, started with `-f json:127.0.0.1`, expects an EXTERNAL physics
simulator: every physics step it sends a binary servo packet (16 PWM values) to
UDP port 9002 and expects a JSON line back with the vehicle state (IMU, position,
attitude, velocity). This node is that external simulator glue for Stonefish:

    Stonefish /odometry + /imu  --->  JSON state  --->  ArduSub SITL (:9002)
    ArduSub SITL servo PWM      --->  [-1,1] setpoints --->  /thruster_setpoints

Then MAVROS / pymavlink / QGroundControl talk to SITL over MAVLink as usual
(e.g. udp:127.0.0.1:14550) - arming, modes (MANUAL/STABILIZE/ALT_HOLD), RC
override, parameters... exactly like the real Pixhawk stack.

Usage:
  1. ros2 launch sauvc_stonefish sauvc_finals.launch.py
  2. sim_vehicle.py -v ArduSub -f json:127.0.0.1 --console   (in ardupilot repo)
     Set FRAME_CONFIG to match your 8-thruster vectored-6DOF frame.
  3. ros2 run sauvc_stonefish ardusub_json_bridge.py  (or python3 .../ardusub_json_bridge.py)
  4. MAVROS: ros2 run mavros mavros_node --ros-args -p fcu_url:=udp://:14550@

NOTES / THINGS TO VERIFY ON YOUR SETUP:
- MOTOR_MAP below maps ArduSub motor outputs (SERVO1..8) to the Stonefish thruster
  order in my_auv.scn: [HFP, HFS, HAP, HAS, VFP, VFS, VAP, VAS]. ArduSub's
  vectored-6DOF motor numbering may differ from this guess - verify against the
  ArduSub motor layout docs and your SERVOx_FUNCTION params, and reorder/flip signs.
- ArduPilot's JSON interface wants accel in body frame INCLUDING gravity (specific
  force, m/s^2), gyro in rad/s body frame, position NED [m], attitude as quaternion
  or euler [rad], velocity NED [m/s]. Stonefish NED matches ArduPilot NED directly.
- Timing: SITL locks its clock to the "timestamp" we send; we use Stonefish
  odometry message time so simulation and SITL stay in lockstep-ish.
"""
import json
import socket
import struct
import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from std_msgs.msg import Float64MultiArray

ROBOT = 'sauvc_auv'          # robot_name used in the scenario
SITL_ADDR = ('127.0.0.1', 9002)
PWM_MIN, PWM_MID, PWM_MAX = 1100, 1500, 1900

# ArduSub SERVO output index (0-based) -> Stonefish thruster index
# Stonefish order (my_auv.scn): 0 HFP, 1 HFS, 2 HAP, 3 HAS, 4 VFP, 5 VFS, 6 VAP, 7 VAS
MOTOR_MAP = [0, 1, 2, 3, 4, 5, 6, 7]   # identity to start - VERIFY AND EDIT
MOTOR_SIGN = [1, 1, 1, 1, 1, 1, 1, 1]  # flip individual motors here


class ArduSubBridge(Node):
    def __init__(self):
        super().__init__('ardusub_json_bridge')
        self.odom = None
        self.imu = None
        self.lock = threading.Lock()

        self.create_subscription(Odometry, f'/{ROBOT}/odometry',
                                 self.odom_cb, qos_profile_sensor_data)
        self.create_subscription(Imu, f'/{ROBOT}/imu',
                                 self.imu_cb, qos_profile_sensor_data)
        self.thr_pub = self.create_publisher(
            Float64MultiArray, f'/{ROBOT}/thruster_setpoints', 10)

        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(SITL_ADDR)
        self.sock.settimeout(1.0)
        threading.Thread(target=self.sitl_loop, daemon=True).start()
        self.get_logger().info(f'Listening for ArduSub SITL on udp:{SITL_ADDR[1]}')

    def odom_cb(self, msg):
        with self.lock:
            self.odom = msg

    def imu_cb(self, msg):
        with self.lock:
            self.imu = msg

    def sitl_loop(self):
        # Servo packet: uint16 magic (18458), uint16 frame_rate, uint32 frame_count, 16x uint16 pwm
        fmt = '<HHI16H'
        size = struct.calcsize(fmt)
        while rclpy.ok():
            try:
                data, addr = self.sock.recvfrom(1024)
            except socket.timeout:
                continue
            if len(data) != size:
                continue
            pkt = struct.unpack(fmt, data)
            if pkt[0] != 18458:
                continue
            pwm = pkt[3:19]

            # PWM -> normalized thruster setpoints
            sp = [0.0] * 8
            for servo_idx, thr_idx in enumerate(MOTOR_MAP):
                v = (pwm[servo_idx] - PWM_MID) / float(PWM_MAX - PWM_MID)
                sp[thr_idx] = max(-1.0, min(1.0, v)) * MOTOR_SIGN[servo_idx]
            m = Float64MultiArray()
            m.data = sp
            self.thr_pub.publish(m)

            # Stonefish state -> JSON reply
            with self.lock:
                odom, imu = self.odom, self.imu
            if odom is None or imu is None:
                continue
            t = odom.header.stamp.sec + odom.header.stamp.nanosec * 1e-9
            q = odom.pose.pose.orientation
            p = odom.pose.pose.position
            v = odom.twist.twist.linear
            state = {
                "timestamp": t,
                "imu": {
                    "gyro": [imu.angular_velocity.x,
                             imu.angular_velocity.y,
                             imu.angular_velocity.z],
                    "accel_body": [imu.linear_acceleration.x,
                                   imu.linear_acceleration.y,
                                   imu.linear_acceleration.z],
                },
                "position": [p.x, p.y, p.z],
                "quaternion": [q.w, q.x, q.y, q.z],
                "velocity": [v.x, v.y, v.z],
            }
            self.sock.sendto((json.dumps(state) + "\n").encode(), addr)


def main():
    rclpy.init()
    node = ArduSubBridge()
    rclpy.spin(node)


if __name__ == '__main__':
    main()
