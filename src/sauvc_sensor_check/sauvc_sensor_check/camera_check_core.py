#!/usr/bin/env python3
"""camera_check_core — Pre-Phase bring-up check for a camera, DRIVER-AGNOSTIC by design.

Switchability: this checks whatever topic is passed to it — sensor_msgs/Image, full
stop. It doesn't know or care whether the publisher is v4l2_camera (a plain USB webcam)
or realsense2_camera (an Intel RealSense). To swap which physical camera serves the
"down" or "front" role, change which driver/launch REMAPS to /camera_down/image_raw or
/camera_front/image_raw (see SETUP.md Pre-Phase) — this script, and everything else in
the workspace that subscribes to those two logical topics, never needs to change.

Reports: resolution, encoding, Hz, mean brightness (lens-cap / bad-exposure sanity
check), works with any encoding cv_bridge can passthrough or convert to mono8/bgr8.
"""
import time
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


def run_camera_check(node_name, default_topic, role_label):
    """Shared implementation; role_label is just a display string ("DOWN"/"FRONT")."""

    class CameraCheckNode(Node):
        def __init__(self):
            super().__init__(node_name)
            self.declare_parameter('topic', default_topic)
            topic = self.get_parameter('topic').value
            self.bridge = CvBridge()
            self.n = 0
            self.t0 = None
            self.last_print = 0.0
            self.create_subscription(Image, topic, self.on_image, 5)
            self.get_logger().info(
                f'[{role_label}] listening on {topic} (driver-agnostic — works with '
                f'v4l2_camera or realsense2_camera, whichever is remapped here) ...')
            self.create_timer(10.0, self.check_silence)

        def check_silence(self):
            if self.n == 0:
                self.get_logger().warn(
                    f'[{role_label}] no frames yet. Checklist: (1) is the camera driver '
                    f'launched (`ros2 topic list` — does the topic exist)? (2) USB '
                    f'connection on a Type-A 3.2 port, not the USB-C debug port? '
                    f'(3) if RealSense: is it on a USB3 port — check `lsusb -t` shows '
                    f'5000M/10000M not 480M? (4) v4l2 permissions — are you in the '
                    f'`video` group (log out/in after usermod)?')

        def on_image(self, msg):
            if self.t0 is None:
                self.t0 = time.time()
            self.n += 1
            now = time.time()
            if now - self.last_print < 1.0:
                return
            self.last_print = now
            elapsed = now - self.t0
            hz = self.n / elapsed if elapsed > 0 else 0.0

            try:
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
                brightness = float(np.mean(cv_img))
                note = ''
                if brightness < 5.0:
                    note = '  <-- near-black frame: lens cap on, no light, or bad exposure?'
                elif brightness > 250.0:
                    note = '  <-- near-white frame: overexposed, or sensor error?'
            except Exception as e:
                brightness = None
                note = f'  <-- cv_bridge conversion failed ({e}); encoding={msg.encoding}'

            b_str = f'{brightness:.1f}/255' if brightness is not None else 'n/a'
            self.get_logger().info(
                f'[{role_label}] {msg.width}x{msg.height} enc={msg.encoding}  '
                f'{hz:.1f} Hz (n={self.n})  mean_brightness={b_str}{note}')

    return CameraCheckNode
