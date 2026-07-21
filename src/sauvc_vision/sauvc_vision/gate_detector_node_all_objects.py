#!/usr/bin/env python3
"""gate_detector_node — Phase 6. Classifies SAUVC props on the forward camera into
DISTINCT NAMED FEATURES with metric relative poses, for visual servoing and for the
landmark/map localization updates in flow_eval_node.

Classification (color + shape + context — no NN needed in the sim):
  orange, tall (aspect>=2)                       -> OrangeFlare      (unique)
  yellow, tall                                   -> FlareYellow      (unique)
  blue:  area>10% of image                       -> WATER, ignored
         tall (aspect>=2)                        -> FlareBlue
         squat (aspect<1.6)                      -> DrumBlue
  green, tall                                    -> GatePostGreen
  red:   tall + a tall green at similar elevation
         and similar height                      -> GatePostRed  (pair => Gate)
         tall, no green partner                  -> FlareRed
         squat                                   -> DrumRed1..3 (ordered by bearing)

Range per feature from KNOWN SIZES (rulebook / scene):
  Gate (both posts):  R = (1.50/2) / tan(dBearing/2)          <- best, width-based
  single tall flare:  R = H / (2 tan(vertExtent/2)),  H: golf flares 0.80 m,
                      orange 1.50 m, gate post 1.00 m (fallback when pair not seen)
  drum:               R = 0.60 / (2 tan(horizExtent/2))
Relative body-FRD position from (R, bearing, elev):  x=R ce cb, y=R ce sb, z=-R se.

Pub:  /vision/features    std_msgs/String  per feature:
                          "name,x,y,z,range,bearing_rad,elev_rad,area_frac"
      /vision/detections  std_msgs/String  legacy "label,bearing,elev,area" (mission)
      /vision/debug_image sensor_msgs/Image overlay
Sub:  image_topic (param)

Terminal: throttled table in the SAME format as landmark_truth_node, so estimated and
true relative poses diff by eye; 'NEW feature' log on first sighting of each name.
A cv2 window (show_detections) draws boxes + labels live.
"""
import math
import time

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from cv_bridge import CvBridge

COLORS = {
    'red':    [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
    'orange': [((10, 130, 90), (22, 255, 255))],
    'yellow': [((23, 100, 100), (35, 255, 255))],
    'green':  [((45, 80, 60), (85, 255, 255))],
    'blue':   [((95, 120, 60), (130, 255, 255))],
}
MIN_AREA_FRAC = 0.0008
WATER_AREA_FRAC = 0.10          # a "blue" blob this big is the water column, not a prop
TALL_ASPECT = 2.0               # bh/bw >= this -> pole-like
SQUAT_ASPECT = 1.6              # bh/bw <  this -> drum-like
GATE_WIDTH_M = 1.50
SIZES_M = {                     # known metric size used for single-blob ranging
    'OrangeFlare': ('h', 1.50), 'FlareRed': ('h', 0.80), 'FlareYellow': ('h', 0.80),
    'FlareBlue': ('h', 0.80), 'GatePostRed': ('h', 1.00), 'GatePostGreen': ('h', 1.00),
    'DrumBlue': ('w', 0.60), 'DrumRed': ('w', 0.60),
}


class GateDetectorNode(Node):
    def __init__(self):
        super().__init__('gate_detector_node')
        p = self.declare_parameter
        p('hfov_deg', 80.0)
        p('vfov_deg', 60.0)
        p('image_topic', '/camera_front/image_raw')
        p('log_period', 1.0)
        p('show_detections', True)      # cv2 window with boxes + labels
        self.bridge = CvBridge()
        self.pub_feat = self.create_publisher(String, '/vision/features', 10)
        self.pub = self.create_publisher(String, '/vision/detections', 10)
        self.pub_dbg = self.create_publisher(Image, '/vision/debug_image', 2)
        topic = self.get_parameter('image_topic').value
        self.create_subscription(Image, topic, self.on_image, 2)
        self._frames = 0
        self._last_log = 0.0
        self._last_frame_wall = None
        self._seen_ever = set()
        self.get_logger().info(
            f'gate_detector up: image_topic={topic}, features on /vision/features, '
            f'cv2 window={"ON" if self.get_parameter("show_detections").value else "off"}')
        self.create_timer(10.0, self._heartbeat)

    def _heartbeat(self):
        if self._last_frame_wall is None:
            self.get_logger().warn('no images yet — sim topic is '
                                   '/sauvc_auv/camera_front/image_color (set image_topic)')
        elif time.time() - self._last_frame_wall > 5.0:
            self.get_logger().warn('image stream STALLED (>5 s)')

    # ---------------- blob extraction ----------------
    def _blobs(self, hsv, w, h):
        out = {}
        for label, ranges in COLORS.items():
            mask = None
            for lo, hi in ranges:
                m = cv2.inRange(hsv, np.array(lo), np.array(hi))
                mask = m if mask is None else cv2.bitwise_or(mask, m)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
            bl = []
            for c in sorted(cnts, key=cv2.contourArea, reverse=True)[:4]:
                area = cv2.contourArea(c) / (w * h)
                if area < MIN_AREA_FRAC:
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                bl.append(dict(area=area, x=x, y=y, bw=bw, bh=bh,
                               cx=x + bw / 2, cy=y + bh / 2,
                               aspect=bh / max(bw, 1)))
            out[label] = bl
        return out

    # ---------------- classification -> named features ----------------
    def _classify(self, blobs):
        feats = {}                       # name -> blob
        tall = lambda b: b['aspect'] >= TALL_ASPECT
        squat = lambda b: b['aspect'] < SQUAT_ASPECT

        for b in blobs['orange']:
            if tall(b):
                feats['OrangeFlare'] = b; break
        for b in blobs['yellow']:
            if tall(b):
                feats['FlareYellow'] = b; break
        for b in blobs['blue']:
            if b['area'] >= WATER_AREA_FRAC:
                continue                 # the water column pretending to be a prop
            if tall(b) and 'FlareBlue' not in feats:
                feats['FlareBlue'] = b
            elif squat(b) and 'DrumBlue' not in feats:
                feats['DrumBlue'] = b
        green_post = next((b for b in blobs['green'] if tall(b)), None)
        if green_post is not None:
            feats['GatePostGreen'] = green_post
        drum_i = 0
        for b in blobs['red']:
            if tall(b):
                if (green_post is not None
                        and abs(b['cy'] - green_post['cy']) < 0.15 * (b['bh'] + green_post['bh'])
                        and 0.5 < b['bh'] / max(green_post['bh'], 1) < 2.0
                        and 'GatePostRed' not in feats):
                    feats['GatePostRed'] = b
                elif 'FlareRed' not in feats:
                    feats['FlareRed'] = b
            elif squat(b) and drum_i < 3:
                drum_i += 1
                feats[f'DrumRed{drum_i}'] = b
        return feats

    # ---------------- ranging + body-frame pose ----------------
    def _pose(self, name, b, w, h, hfov, vfov, feats):
        bearing = ((b['cx'] / w) - 0.5) * hfov
        elev = -((b['cy'] / h) - 0.5) * vfov          # + up
        rng = None
        if name.startswith('GatePost') and 'GatePostRed' in feats and 'GatePostGreen' in feats:
            b1, b2 = feats['GatePostRed'], feats['GatePostGreen']
            dth = abs((b1['cx'] - b2['cx']) / w) * hfov
            if dth > 1e-3:
                rng = (GATE_WIDTH_M / 2.0) / math.tan(dth / 2.0)
        if rng is None:
            key = 'DrumRed' if name.startswith('DrumRed') else name
            dim, size = SIZES_M.get(key, ('h', 0.8))
            ext = (b['bh'] / h) * vfov if dim == 'h' else (b['bw'] / w) * hfov
            if ext > 1e-3:
                rng = size / (2.0 * math.tan(ext / 2.0))
        if rng is None or rng > 40.0:
            return None
        ce, se = math.cos(elev), math.sin(elev)
        cb, sb = math.cos(bearing), math.sin(bearing)
        # body FRD (camera assumed forward-aligned): z down => z = -R sin(elev_up)
        return dict(x=rng * ce * cb, y=rng * ce * sb, z=-rng * se,
                    range=rng, brg=bearing, elev=elev, area=b['area'])

    # ---------------- per frame ----------------
    def on_image(self, msg):
        self._last_frame_wall = time.time()
        self._frames += 1
        bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        h, w = bgr.shape[:2]
        hfov = math.radians(self.get_parameter('hfov_deg').value)
        vfov = math.radians(self.get_parameter('vfov_deg').value)
        hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
        feats = self._classify(self._blobs(hsv, w, h))
        dbg = bgr.copy()

        poses = {}
        for name, b in feats.items():
            pz = self._pose(name, b, w, h, hfov, vfov, feats)
            if pz is None:
                continue
            poses[name] = pz
            self.pub_feat.publish(String(data=(
                f"{name},{pz['x']:.3f},{pz['y']:.3f},{pz['z']:.3f},"
                f"{pz['range']:.3f},{pz['brg']:.5f},{pz['elev']:.5f},{pz['area']:.5f}")))
            self.pub.publish(String(data=(
                f"{name},{pz['brg']:.4f},{pz['elev']:.4f},{pz['area']:.5f}")))
            if name not in self._seen_ever:
                self._seen_ever.add(name)
                self.get_logger().info(
                    f"NEW feature first sighted: {name} at range {pz['range']:.2f} m, "
                    f"bearing {math.degrees(pz['brg']):+.1f} deg — first-fix quality "
                    "matters: this is what the map freezes.")
            cv2.rectangle(dbg, (b['x'], b['y']),
                          (b['x'] + b['bw'], b['y'] + b['bh']), (0, 255, 0), 2)
            cv2.putText(dbg, f"{name} {pz['range']:.1f}m",
                        (b['x'], max(b['y'] - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if poses and t - self._last_log >= self.get_parameter('log_period').value:
            self._last_log = t
            print('\n─ ESTIMATED feature poses in body FRD (x fwd, y stbd, z down) ─'
                  ' bearing +stbd ─')
            print(f"  {'feature':<16}{'x':>8}{'y':>8}{'z':>8}"
                  f"{'range':>8}{'brg°':>8}{'elev°':>8}")
            for name in sorted(poses):
                pz = poses[name]
                print(f"  {name:<16}{pz['x']:>+8.2f}{pz['y']:>+8.2f}{pz['z']:>+8.2f}"
                      f"{pz['range']:>8.2f}{math.degrees(pz['brg']):>+8.1f}"
                      f"{-math.degrees(pz['elev']):>+8.1f}")

        self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8'))
        if self.get_parameter('show_detections').value:
            cv2.imshow('gate_detector', dbg)
            cv2.waitKey(1)


def main():
    rclpy.init()
    try:
        rclpy.spin(GateDetectorNode())
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
