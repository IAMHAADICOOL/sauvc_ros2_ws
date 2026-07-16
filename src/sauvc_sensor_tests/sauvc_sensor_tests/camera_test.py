#!/usr/bin/env python3
"""Camera test: opens two OpenCV windows showing the front and downward cameras.

Run:  ros2 run sauvc_sensor_tests camera_test
Params: front_topic (default /sauvc_auv/camera_front),
        down_topic  (default /sauvc_auv/camera_down)
Press q in either window to quit.

Note: stonefish_ros2 publishes the image on <topic>/image_color and the intrinsics
on <topic>/camera_info; this node subscribes to <topic>/image_color and falls back
to the bare <topic> name if your version publishes there instead.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2


class CameraTest(Node):
    def __init__(self):
        super().__init__('camera_test')
        self.declare_parameter('front_topic', '/sauvc_auv/camera_front')
        self.declare_parameter('down_topic', '/sauvc_auv/camera_down')
        self.bridge = CvBridge()
        self.frames = {}
        for key in ('front', 'down'):
            base = self.get_parameter(f'{key}_topic').value
            # subscribe to both candidate names; whichever publishes wins
            for topic in (base + '/image_color', base):
                self.create_subscription(
                    Image, topic,
                    lambda msg, k=key: self.cb(msg, k),
                    qos_profile_sensor_data)
        self.timer = self.create_timer(0.03, self.show)
        self.get_logger().info('waiting for images... (press q in a window to quit)')

    def cb(self, msg: Image, key: str):
        self.frames[key] = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

    def show(self):
        for key, frame in self.frames.items():
            cv2.imshow(f'{key} camera', frame)
        if self.frames and (cv2.waitKey(1) & 0xFF) == ord('q'):
            rclpy.shutdown()


def main():
    rclpy.init()
    node = CameraTest()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, rclpy.executors.ExternalShutdownException):
        pass
    finally:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
