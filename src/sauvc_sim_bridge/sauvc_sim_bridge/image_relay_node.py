#!/usr/bin/env python3
"""image_relay_node — republishes Stonefish camera topics under the names the real nodes expect.

Sub: /sauvc_auv/camera_down/image_color   sensor_msgs/Image
     /sauvc_auv/camera_front/image_color  sensor_msgs/Image
Pub: /camera_down/image_raw               sensor_msgs/Image
     /camera_front/image_raw              sensor_msgs/Image

WHY A RELAY INSTEAD OF LAUNCH REMAPS
------------------------------------
Three real nodes consume camera images, and they DIFFER in how the topic is set:

    flow_velocity_node : image_topic PARAMETER, default /camera_down/image_raw
    lane_heading_node  : HARDCODED   /camera_down/image_raw   (no parameter)
    gate_detector_node : HARDCODED   /camera_front/image_raw  (no parameter)

Because two of the three hardcode the name, a per-node launch remapping would fix
flow_velocity_node but not the other two, and editing working hardware nodes just to add
a parameter is the wrong trade. A single relay that makes `/camera_*/image_raw` exist —
carrying exactly the sim's pixels — fixes all three at once and leaves every real node
untouched. It is the camera analogue of what imu_shim/depth_shim do for the other sensors.

The relay is a straight passthrough: same Image message, same header (including the
wall-clock stamp Stonefish sets), only the topic name changes. No rethrottling, no
re-encoding. stonefish_ros2 publishes rgb8; every consumer asks cv_bridge for mono8 or
bgr8, both of which convert from rgb8 cleanly, so no colour conversion is needed here.

A note on the camera rate you measured (~9.5 Hz down, ~7.5 Hz front, vs the declared
30): this relay does NOT change that — it forwards whatever arrives. The frame rate is a
rendering-cost problem in Stonefish, addressed in the scene (drop resolution to 640x480)
or the launch file (rendering_quality), not here. The relay only renames.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image


class ImageRelayNode(Node):
    def __init__(self):
        super().__init__('image_relay_node')
        # pairs of (source image_color topic, destination image_raw topic)
        self.declare_parameter('down_src', '/sauvc_auv/camera_down/image_color')
        self.declare_parameter('down_dst', '/camera_down/image_raw')
        self.declare_parameter('front_src', '/sauvc_auv/camera_front/image_color')
        self.declare_parameter('front_dst', '/camera_front/image_raw')
        g = lambda n: self.get_parameter(n).value

        self._wire(g('down_src'), g('down_dst'))
        self._wire(g('front_src'), g('front_dst'))
        self.get_logger().info(
            'image_relay: /sauvc_auv/*/image_color -> /camera_*/image_raw (passthrough)')

    def _wire(self, src, dst):
        pub = self.create_publisher(Image, dst, qos_profile_sensor_data)
        # default-arg binding so the closure captures THIS pub, not the last one
        self.create_subscription(
            Image, src, lambda msg, p=pub: p.publish(msg), qos_profile_sensor_data)


def main():
    rclpy.init()
    rclpy.spin(ImageRelayNode())


if __name__ == '__main__':
    main()
