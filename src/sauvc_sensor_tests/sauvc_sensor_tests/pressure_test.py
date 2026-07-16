#!/usr/bin/env python3
"""Pressure sensor test: prints pressure [Pa] and derived depth [m] in the terminal.

Depth is derived as (p - p_ref) / (rho * g). The first sample is taken as the
surface reference (the vehicle spawns at/near the surface); pass a ROS param
`p_ref` to override, e.g. 101325.0 for absolute-including-atmosphere sensors.

Run:  ros2 run sauvc_sensor_tests pressure_test
      ros2 run sauvc_sensor_tests pressure_test --ros-args -p topic:=/sauvc_auv/pressure
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import FluidPressure

RHO = 1000.0  # fresh pool water [kg/m^3]
G = 9.81


class PressureTest(Node):
    def __init__(self):
        super().__init__('pressure_test')
        self.declare_parameter('topic', '/sauvc_auv/pressure')
        self.declare_parameter('p_ref', float('nan'))
        self.p_ref = self.get_parameter('p_ref').value
        topic = self.get_parameter('topic').value
        self.sub = self.create_subscription(
            FluidPressure, topic, self.cb, qos_profile_sensor_data)
        self.get_logger().info(f'listening on {topic}')

    def cb(self, msg: FluidPressure):
        p = msg.fluid_pressure
        if self.p_ref != self.p_ref:  # NaN -> auto-zero on first sample
            self.p_ref = p
            self.get_logger().info(f'surface reference set: {p:.1f} Pa')
        depth = (p - self.p_ref) / (RHO * G)
        print(f'\rpressure: {p:12.2f} Pa   depth: {depth:+7.3f} m   '
              f'(variance {msg.variance:.2f})', end='', flush=True)


def main():
    rclpy.init()
    rclpy.spin(PressureTest())


if __name__ == '__main__':
    main()
