#!/usr/bin/env python3
"""image_saver_node (sauvc_vision) — dataset collection while teleoperating.

Saves throttled JPGs from any Image topic into a timestamped session directory,
ready for annotation (YOLO). Run it alongside teleop; drive varied approaches to
each prop: near/far, off-axis, partially out of frame, near the surface glare.

  ros2 run sauvc_vision image_saver_node --ros-args \
      -p image_topic:=/sauvc_auv/camera_front/image_color -p rate_hz:=2.0
  # real vehicle: -p image_topic:=/camera_front/image_raw

Params: image_topic (str), rate_hz (double, default 2.0 — saving faster than ~3 Hz
just yields near-duplicate frames), out_dir (default ~/sauvc_dataset), prefix (str).
Output: <out_dir>/<prefix>_<YYYYmmdd_HHMMSS>/frame_000123.jpg ; count logged every 50.
"""
import os
import time
from datetime import datetime

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class ImageSaverNode(Node):
    def __init__(self):
        super().__init__('image_saver_node')
        self.declare_parameter('image_topic', '/camera_front/image_raw')
        self.declare_parameter('rate_hz', 2.0)
        self.declare_parameter('out_dir', os.path.expanduser('~/sauvc_dataset'))
        self.declare_parameter('prefix', 'session')
        g = lambda n: self.get_parameter(n).value
        self.period = 1.0 / max(g('rate_hz'), 0.1)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.dir = os.path.join(os.path.expanduser(g('out_dir')),
                                f"{g('prefix')}_{stamp}")
        os.makedirs(self.dir, exist_ok=True)
        self.bridge = CvBridge()
        self.n = 0
        self._last = 0.0
        self.create_subscription(Image, g('image_topic'), self.on_image,
                                 qos_profile_sensor_data)
        self.get_logger().info(f"saving {g('rate_hz')} Hz from {g('image_topic')} "
                               f"-> {self.dir}")

    def on_image(self, msg):
        now = time.time()
        if now - self._last < self.period:
            return
        self._last = now
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        cv2.imwrite(os.path.join(self.dir, f'frame_{self.n:06d}.jpg'), bgr)
        self.n += 1
        if self.n % 50 == 0:
            self.get_logger().info(f'{self.n} frames saved')


def main():
    rclpy.init()
    try:
        rclpy.spin(ImageSaverNode())
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
