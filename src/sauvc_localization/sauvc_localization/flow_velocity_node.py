#!/usr/bin/env python3
"""flow_velocity_node — publishes body-frame vx, vy from downward-cam optical flow.

Pub:  /flow/twist   geometry_msgs/TwistWithCovarianceStamped
Sub:  image_topic (param), /imu/data, /altitude
See flow_core.FlowVelocityEstimator for the math. Phase 3 of the pipeline.
"""
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Float32
from geometry_msgs.msg import TwistWithCovarianceStamped
from cv_bridge import CvBridge

from sauvc_localization.flow_core import FlowVelocityEstimator


class FlowVelocityNode(Node):
    def __init__(self):
        super().__init__('flow_velocity_node')
        self.declare_parameter('fx', 700.0)
        self.declare_parameter('fy', 700.0)
        self.declare_parameter('cx', 320.0)
        self.declare_parameter('cy', 240.0)
        self.declare_parameter('swap_xy', False)
        self.declare_parameter('sign_x', 1.0)
        self.declare_parameter('sign_y', 1.0)
        self.declare_parameter('base_var', 0.02)
        self.declare_parameter('image_topic', '/camera_down/image_raw')
        g = lambda n: self.get_parameter(n).value

        self.est = FlowVelocityEstimator(g('fx'), g('fy'), g('cx'), g('cy'),
                                         swap_xy=g('swap_xy'),
                                         sign_x=g('sign_x'), sign_y=g('sign_y'))
        self.base_var = g('base_var')
        self.bridge = CvBridge()
        self.gyro = (0.0, 0.0, 0.0)
        self.altitude = None
        self.last_stamp = None

        self.pub = self.create_publisher(TwistWithCovarianceStamped, '/flow/twist', 10)
        self.create_subscription(Imu, '/imu/data', self.on_imu, 50)
        self.create_subscription(Float32, '/altitude', self.on_alt, 10)
        self.create_subscription(Image, g('image_topic'), self.on_image, 5)
        self.get_logger().info('flow_velocity_node up, waiting for images + altitude')

    def on_imu(self, msg):
        self.gyro = (msg.angular_velocity.x, msg.angular_velocity.y,
                     msg.angular_velocity.z)

    def on_alt(self, msg):
        self.altitude = msg.data

    def on_image(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dt = 0.0 if self.last_stamp is None else (t - self.last_stamp)
        self.last_stamp = t
        gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')

        # Body gyro -> camera frame, same convention as the velocity mapping:
        # camera x (image right) = -body y ; camera y (image down) = -body x
        gx_cam = -self.gyro[1]
        gy_cam = -self.gyro[0]

        out = self.est.process(gray, dt, (gx_cam, gy_cam), self.altitude)
        if out is None:
            return
        m = TwistWithCovarianceStamped()
        m.header.stamp = msg.header.stamp
        m.header.frame_id = 'base_link'
        m.twist.twist.linear.x = out['vx']
        m.twist.twist.linear.y = out['vy']
        q = max(out['n_inliers'], 1)
        var = self.base_var * (1.0 + out['spread_px']) * (100.0 / q)
        cov = [0.0] * 36
        cov[0] = var
        cov[7] = var
        cov[14] = 1e6
        m.twist.covariance = cov
        self.pub.publish(m)


def main():
    rclpy.init()
    rclpy.spin(FlowVelocityNode())


if __name__ == '__main__':
    main()
