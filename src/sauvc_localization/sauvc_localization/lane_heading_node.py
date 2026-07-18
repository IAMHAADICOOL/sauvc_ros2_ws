#!/usr/bin/env python3
"""
lane_heading_node.py — Phase 4. Absolute heading from pool floor lines (mod 90°).

Idea: competition pools have lane lines / tile grout aligned with the pool axes. The
dominant line direction in the DOWN camera gives vehicle yaw relative to the pool,
modulo 90°. Fused as a slow complementary correction to the gyro-integrated yaw, this
cancels drift with no magnetometer. The mod-90 ambiguity is fine because gyro yaw never
drifts ~45° between corrections.

Subscribes: /camera_down/image_raw (sensor_msgs/Image)
            /imu/data              (sensor_msgs/Imu — uses orientation yaw as the fast source)
Publishes : /heading/pool_relative (std_msgs/Float32, corrected yaw [rad], pool-axis frame)
            /heading/line_meas     (std_msgs/Float32, raw line angle [rad], debug)

Set `pool_axis_offset` so that yaw=0 points along your chosen mission axis (e.g. from
start zone toward the gate). Determine it once at the venue: hold the vehicle pointing
at the gate, read /heading/line_meas, put that value in the parameter.
"""

import math
import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, Imu
from std_msgs.msg import Float32
from cv_bridge import CvBridge


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_from_quat(q):
    return math.atan2(2.0 * (q.w * q.z + q.x * q.y),
                      1.0 - 2.0 * (q.y * q.y + q.z * q.z))


class LaneHeadingNode(Node):
    def __init__(self):
        super().__init__('lane_heading_node')
        self.declare_parameter('pool_axis_offset', 0.0)   # rad
        self.declare_parameter('gain', 0.02)              # complementary gain per frame
        self.declare_parameter('min_lines', 4)
        self.bridge = CvBridge()

        self.gyro_yaw = None          # fast source: IMU yaw (drifts)
        self.offset = 0.0             # slow correction estimated from lines
        self.pub_yaw = self.create_publisher(Float32, '/heading/pool_relative', 10)
        self.pub_meas = self.create_publisher(Float32, '/heading/line_meas', 10)
        self.create_subscription(Imu, '/imu/data', self.on_imu, 50)
        self.create_subscription(Image, '/camera_down/image_raw', self.on_image, 5)

    def on_imu(self, msg):
        self.gyro_yaw = yaw_from_quat(msg.orientation)
        if self.gyro_yaw is not None:
            out = Float32()
            out.data = wrap(self.gyro_yaw + self.offset
                            - self.get_parameter('pool_axis_offset').value)
            self.pub_yaw.publish(out)

    def detect_line_angle(self, gray):
        """Dominant floor-line direction in IMAGE frame, in (-pi/4, pi/4]. None if unsure."""
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(gray, 40, 120)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60,
                                minLineLength=gray.shape[1] // 5, maxLineGap=12)
        if lines is None or len(lines) < self.get_parameter('min_lines').value:
            return None
        # Fold every segment angle into mod-90° space and take a length-weighted
        # circular mean (period pi/2 -> multiply angles by 4).
        s = c = 0.0
        for (x1, y1, x2, y2) in lines[:, 0]:
            ang = math.atan2(y2 - y1, x2 - x1)          # (-pi, pi]
            L = math.hypot(x2 - x1, y2 - y1)
            s += L * math.sin(4 * ang)
            c += L * math.cos(4 * ang)
        if s == 0 and c == 0:
            return None
        mean4 = math.atan2(s, c)
        # concentration check: reject frames where line directions disagree wildly
        R = math.hypot(s, c) / sum(math.hypot(x2 - x1, y2 - y1)
                                   for (x1, y1, x2, y2) in lines[:, 0])
        if R < 0.6:
            return None
        return mean4 / 4.0                               # (-pi/4, pi/4]

    def on_image(self, msg):
        if self.gyro_yaw is None:
            return
        gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding='mono8')
        ang = self.detect_line_angle(gray)
        if ang is None:
            return
        self.pub_meas.publish(Float32(data=float(ang)))

        # Lines at image angle `ang` mean vehicle yaw relative to the pool grid is -ang
        # (mod 90°). Pick the mod-90 branch closest to the current corrected yaw.
        cur = wrap(self.gyro_yaw + self.offset)
        meas = -ang
        k = round((cur - meas) / (math.pi / 2))
        meas_unwrapped = meas + k * (math.pi / 2)
        err = wrap(meas_unwrapped - cur)
        if abs(err) < math.radians(20):                  # sanity gate
            self.offset = wrap(self.offset + self.get_parameter('gain').value * err)


def main():
    rclpy.init()
    rclpy.spin(LaneHeadingNode())


if __name__ == '__main__':
    main()
