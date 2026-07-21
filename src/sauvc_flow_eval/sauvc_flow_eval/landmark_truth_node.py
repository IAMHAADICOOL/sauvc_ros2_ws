#!/usr/bin/env python3
"""landmark_truth_node — SIM-ONLY ground truth for landmark-relative poses.

The sim defines everything, so the true pose of every prop w.r.t. the vehicle is
computable exactly: parse the scene for landmark world positions, subscribe to
ground-truth odometry, rotate (landmark − vehicle) into the vehicle BODY frame.
Use it to score any gate/flare/drum detector: your estimator's relative pose vs
these numbers IS the detection error, with no other error source mixed in.

Publishes, per landmark:
  /truth/rel/<name>       geometry_msgs/PointStamped   position in body FRD [m]
and prints a throttled table: body x/y/z, range, bearing, elevation.

Frames: scene + odometry are NED world; body frame here is FRD (x fwd, y stbd,
z down) — the same convention a forward-camera detector naturally reports in
(bearing = atan2(y_frd, x_frd), positive to starboard; elevation positive DOWN).

Run (standalone, or add a console_scripts entry point):
  python3 landmark_truth_node.py --ros-args \
      -p scene_file:=$HOME/Robotics_Job/sauvc_ws/src/sauvc_stonefish/scenarios/sauvc_finals.scn
  # randomized arena: point scene_file at the generated scene in /tmp (printed at launch)
  ros2 run sauvc_flow_eval landmark_truth_node --ros-args -p scene_file:=/tmp/<generated>.scn

Parameters:
  scene_file   REQUIRED. The .scn to parse (statics + dynamics named gate/flare/drum/tub).
  robot_name   default 'sauvc_auv' — ground-truth odometry topic namespace.
  print_rate   default 2.0 Hz — table throttle. 0 disables printing (topics only).
  only         default '' — comma-separated name filter, e.g. 'GateCenter,OrangeFlare'.

NOTE on dynamics: flare/golf-ball positions are parsed from the SCENE (spawn pose).
After the vehicle bumps one, its true pose changes and Stonefish does not publish it;
the printed value for that prop is then stale — fine for pre-contact detector scoring,
which is when the pose matters.
"""
import math
import re
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PointStamped


def parse_scene(path):
    root = ET.fromstring(open(path).read())
    pat = re.compile(r'gate|flare|drum|tub', re.I)
    landmarks = {}
    for tag in ('static', 'dynamic'):
        for el in root.iter(tag):
            name = el.get('name', '')
            wt = el.find('world_transform')
            if name and pat.search(name) and wt is not None:
                landmarks[name] = np.array(
                    [float(v) for v in wt.get('xyz').split()])
    if 'GatePostPort' in landmarks and 'GatePostStbd' in landmarks:
        landmarks['GateCenter'] = 0.5 * (landmarks['GatePostPort']
                                         + landmarks['GatePostStbd'])
    return landmarks


def quat_to_R(x, y, z, w):
    """Body->world rotation matrix from a (x,y,z,w) quaternion."""
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]])


class LandmarkTruthNode(Node):
    def __init__(self):
        super().__init__('landmark_truth_node')
        self.declare_parameter('scene_file', '')
        self.declare_parameter('robot_name', 'sauvc_auv')
        self.declare_parameter('print_rate', 2.0)
        self.declare_parameter('only', '')
        g = lambda n: self.get_parameter(n).value

        sf = g('scene_file')
        if not sf:
            raise RuntimeError('scene_file parameter is required')
        self.landmarks = parse_scene(sf)
        only = [s.strip() for s in g('only').split(',') if s.strip()]
        if only:
            self.landmarks = {k: v for k, v in self.landmarks.items() if k in only}
        if not self.landmarks:
            raise RuntimeError('no landmarks parsed (check scene_file / only filter)')
        self.get_logger().info(
            f'{len(self.landmarks)} landmarks: ' + ', '.join(sorted(self.landmarks)))

        self.pubs = {name: self.create_publisher(PointStamped, f'/truth/rel/{name}', 10)
                     for name in self.landmarks}
        self.print_period = (1.0 / g('print_rate')) if g('print_rate') > 0 else None
        self._last_print = 0.0
        self.create_subscription(Odometry, f"/{g('robot_name')}/odometry",
                                 self.on_odom, qos_profile_sensor_data)

    def on_odom(self, msg):
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        pos = np.array([p.x, p.y, p.z])                # NED world
        Rbw = quat_to_R(q.x, q.y, q.z, q.w)            # body(FRD)->world(NED)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        rows = []
        for name in sorted(self.landmarks):
            rel_w = self.landmarks[name] - pos          # world NED
            rel_b = Rbw.T @ rel_w                       # body FRD
            m = PointStamped()
            m.header = msg.header
            m.header.frame_id = 'base_link_frd'
            m.point.x, m.point.y, m.point.z = (float(rel_b[0]), float(rel_b[1]),
                                               float(rel_b[2]))
            self.pubs[name].publish(m)
            rng = float(np.linalg.norm(rel_b))
            brg = math.degrees(math.atan2(rel_b[1], rel_b[0]))
            elv = math.degrees(math.atan2(rel_b[2], math.hypot(rel_b[0], rel_b[1])))
            rows.append((name, rel_b, rng, brg, elv))

        if self.print_period is not None and t - self._last_print >= self.print_period:
            self._last_print = t
            print(f"\n─ TRUE landmark poses in body FRD "
                  f"(x fwd, y stbd, z down) ─ bearing +stbd, elev +down ─")
            print(f"  {'landmark':<16}{'x':>8}{'y':>8}{'z':>8}"
                  f"{'range':>8}{'brg°':>8}{'elev°':>8}")
            for name, rb, rng, brg, elv in rows:
                print(f"  {name:<16}{rb[0]:>+8.2f}{rb[1]:>+8.2f}{rb[2]:>+8.2f}"
                      f"{rng:>8.2f}{brg:>+8.1f}{elv:>+8.1f}")


def main():
    rclpy.init()
    node = LandmarkTruthNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
