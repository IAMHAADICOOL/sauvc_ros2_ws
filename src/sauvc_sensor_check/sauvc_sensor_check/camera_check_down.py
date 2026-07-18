#!/usr/bin/env python3
"""camera_check_down — Pre-Phase bring-up check for whatever is publishing
/camera_down/image_raw (v4l2 webcam or RealSense, remapped — see camera_check_core.py).

Usage:
  ros2 run sauvc_sensor_check camera_check_down --ros-args -p topic:=/camera_down/image_raw
"""
import rclpy
from sauvc_sensor_check.camera_check_core import run_camera_check


def main():
    rclpy.init()
    NodeCls = run_camera_check('camera_check_down', '/camera_down/image_raw', 'DOWN')
    node = NodeCls()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
