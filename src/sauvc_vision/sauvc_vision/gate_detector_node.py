#!/usr/bin/env python3
"""gate_detector_node — Phase 6 starter. Color-blob detection of SAUVC props on the
forward camera, publishing bearing + rough size for visual servoing and landmark resets.

This is a deliberately simple HSV color detector to get you moving; upgrade to a small
CNN (YOLO on the Jetson) later if pool testing shows color thresholds are too fragile
under your venue's lighting.

Pub:  /vision/detections   std_msgs/String  "label,bearing_rad,elev_rad,area_frac"
      /vision/debug_image  sensor_msgs/Image (thresholded overlay, for rqt_image_view)
Sub:  /camera_front/image_raw

Targets and default HSV ranges (TUNE THESE IN YOUR POOL — Phase 6 test 1):
  gate_red / gate_green : striped gate side markings (Navigation task)
  flare_orange          : the AVOID flare (Navigation task)
  flare_red/yellow/blue : Communication & Localization task flares
"""
import math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

# label: (lowHSV, highHSV, min_area_frac). Red needs two ranges (hue wraps at 180).
COLORS = {
    'red':    [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
    'orange': [((10, 130, 90), (22, 255, 255))],
    'yellow': [((23, 100, 100), (35, 255, 255))],
    'green':  [((45, 80, 60), (85, 255, 255))],
    'blue':   [((95, 120, 60), (130, 255, 255))],
}
MIN_AREA_FRAC = 0.0015   # ignore blobs smaller than this fraction of the image


class GateDetectorNode(Node):
    def __init__(self):
        super().__init__('gate_detector_node')
        self.declare_parameter('hfov_deg', 80.0)   # horizontal field of view underwater
        self.declare_parameter('vfov_deg', 60.0)
        self.bridge = CvBridge()
        self.pub = self.create_publisher(String, '/vision/detections', 10)
        self.pub_dbg = self.create_publisher(Image, '/vision/debug_image', 2)
        self.create_subscription(Image, '/camera_front/image_raw', self.on_image, 2)

    def on_image(self, msg):
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = bgr.shape[:2]
        hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
        hfov = math.radians(self.get_parameter('hfov_deg').value)
        vfov = math.radians(self.get_parameter('vfov_deg').value)
        dbg = bgr.copy()

        for label, ranges in COLORS.items():
            mask = None
            for lo, hi in ranges:
                m = cv2.inRange(hsv, np.array(lo), np.array(hi))
                mask = m if mask is None else cv2.bitwise_or(mask, m)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            area = cv2.contourArea(c) / (w * h)
            if area < MIN_AREA_FRAC:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            cx, cy = x + bw / 2, y + bh / 2
            bearing = ((cx / w) - 0.5) * hfov          # + = target to the right
            elev = -((cy / h) - 0.5) * vfov            # + = target above center
            self.pub.publish(String(data=f'{label},{bearing:.4f},{elev:.4f},{area:.5f}'))
            cv2.rectangle(dbg, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(dbg, label, (x, y - 6), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 0), 2)

        self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8'))


def main():
    rclpy.init()
    rclpy.spin(GateDetectorNode())


if __name__ == '__main__':
    main()
