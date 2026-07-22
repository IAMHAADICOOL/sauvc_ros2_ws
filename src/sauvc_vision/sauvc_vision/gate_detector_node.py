#!/usr/bin/env python3
"""gate_detector_node — Phase 6. Classifies SAUVC props into DISTINCT NAMED FEATURES
with metric relative poses, for visual servoing and the landmark updates in
flow_eval_node.

TWO detection backends, same downstream math:
  * HSV color+shape+context (default, sim-tuned)   — the original pipeline
  * YOLO (`use_yolo:=true yolo_model:=/path/best.pt`) — your trained model; boxes
    replace the color blobs, class names map to feature names, the SAME known-size
    ranging (_pose) turns each box into a metric body-FRD position. The gate
    pair-width ranging still applies when both posts are detected.

GROUND-TRUTH ERROR COLUMNS (new): this node now also subscribes to the
/truth/rel/<name> topics that landmark_truth_node publishes and prints, next to
every estimated feature pose, the deviation from truth (dx dy dz dR dbrg delev).
Run landmark_truth_node alongside; without it the error columns show '--'.
Detector names differ from scene names (GatePostRed vs GatePostPort): the
`truth_name_map` parameter holds explicit pairs; anything unmapped is matched to
the NEAREST truth landmark of the same class (drum/flare/gate) so DrumRed1..3
(ordered by bearing) still score against DrumRed2/DrumRed3/DrumRedPinger.

Pub:  /vision/features    std_msgs/String  "name,x,y,z,range,bearing_rad,elev_rad,area_frac"
      /vision/detections  std_msgs/String  legacy "label,bearing,elev,area"
      /vision/debug_image sensor_msgs/Image overlay
Sub:  image_topic (param), /truth/rel/* (auto-discovered)
"""
import math
import time

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import PointStamped
from cv_bridge import CvBridge

COLORS = {
    'red':    [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
    'orange': [((10, 130, 90), (22, 255, 255))],
    'yellow': [((23, 100, 100), (35, 255, 255))],
    'green':  [((45, 80, 60), (85, 255, 255))],
    'blue':   [((95, 120, 60), (130, 255, 255))],
}
MIN_AREA_FRAC = 0.0008
WATER_AREA_FRAC = 0.10
TALL_ASPECT = 2.0
SQUAT_ASPECT = 1.6
GATE_WIDTH_M = 1.50
SIZES_M = {
    'OrangeFlare': ('h', 1.50), 'FlareRed': ('h', 0.80), 'FlareYellow': ('h', 0.80),
    'FlareBlue': ('h', 0.80), 'GatePostRed': ('h', 1.00), 'GatePostGreen': ('h', 1.00),
    'DrumBlue': ('w', 0.60), 'DrumRed': ('w', 0.60),
}
# default detector-name -> truth-name pairs (scene: sauvc_finals.scn)
DEFAULT_TRUTH_MAP = ('GatePostRed=GatePostPort,GatePostGreen=GatePostStbd')
# class token used for nearest-truth fallback matching
def _class_token(name):
    for tok in ('GatePost', 'DrumRed', 'DrumBlue', 'Drum', 'FlareRed', 'FlareYellow',
                'FlareBlue', 'OrangeFlare', 'Flare', 'Gate'):
        if name.startswith(tok):
            return tok
    return name


class GateDetectorNode(Node):
    def __init__(self):
        super().__init__('gate_detector_node')
        p = self.declare_parameter
        p('hfov_deg', 80.0)
        p('vfov_deg', 60.0)
        p('image_topic', '/camera_front/image_raw')
        p('log_period', 1.0)
        p('show_detections', True)
        # --- YOLO backend ---
        p('use_yolo', False)
        p('yolo_model', '')              # path to your trained .pt
        p('yolo_conf', 0.4)
        # 'model_class=FeatureName,...'; empty -> auto-match on lowercase tokens
        p('yolo_class_map', '')
        # --- truth comparison ---
        p('truth_name_map', DEFAULT_TRUTH_MAP)
        p('truth_max_age', 1.0)          # s; older cached truth prints as '--'
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

        # truth cache: name -> (t_wall, np.array([x,y,z]) body FRD)
        self._truth = {}
        self._truth_subs = {}
        self._truth_map = {}
        for pair in self.get_parameter('truth_name_map').value.split(','):
            if '=' in pair:
                a, b = pair.split('=', 1)
                self._truth_map[a.strip()] = b.strip()
        self.create_timer(2.0, self._discover_truth_topics)

        # YOLO (lazy, optional)
        self._yolo = None
        self._yolo_names = {}
        if self.get_parameter('use_yolo').value:
            self._load_yolo()

        self.get_logger().info(
            f'gate_detector up: image_topic={topic}, backend='
            + ('YOLO' if self._yolo is not None else 'HSV')
            + ', features on /vision/features')
        self.create_timer(10.0, self._heartbeat)

    # ---------------- YOLO backend ----------------
    def _load_yolo(self):
        path = self.get_parameter('yolo_model').value
        try:
            from ultralytics import YOLO
            self._yolo = YOLO(path)
            cmap = {}
            raw = self.get_parameter('yolo_class_map').value
            for pair in raw.split(','):
                if '=' in pair:
                    a, b = pair.split('=', 1)
                    cmap[a.strip().lower()] = b.strip()
            self._yolo_names = cmap
            self.get_logger().info(f'YOLO model loaded: {path}, '
                                   f'classes={list(self._yolo.names.values())}')
        except Exception as e:
            self._yolo = None
            self.get_logger().error(f'YOLO unavailable ({e}) — falling back to HSV')

    def _yolo_feature_name(self, cls_name):
        """Map a model class to a canonical feature name."""
        if cls_name.lower() in self._yolo_names:
            return self._yolo_names[cls_name.lower()]
        c = cls_name.lower().replace('_', '').replace('-', '')
        table = {'gatepostred': 'GatePostRed', 'redgatepost': 'GatePostRed',
                 'gatepostgreen': 'GatePostGreen', 'greengatepost': 'GatePostGreen',
                 'gate': 'GatePostRed',
                 'orangeflare': 'OrangeFlare', 'flareorange': 'OrangeFlare',
                 'redflare': 'FlareRed', 'flarered': 'FlareRed',
                 'yellowflare': 'FlareYellow', 'flareyellow': 'FlareYellow',
                 'blueflare': 'FlareBlue', 'flareblue': 'FlareBlue',
                 'bluedrum': 'DrumBlue', 'drumblue': 'DrumBlue',
                 'reddrum': 'DrumRed', 'drumred': 'DrumRed', 'drum': 'DrumRed'}
        return table.get(c)

    def _classify_yolo(self, bgr, w, h):
        conf = self.get_parameter('yolo_conf').value
        res = self._yolo(bgr, conf=conf, verbose=False)[0]
        feats = {}
        red_drums = []
        for box in res.boxes:
            cls_name = self._yolo.names[int(box.cls[0])]
            base = self._yolo_feature_name(cls_name)
            if base is None:
                continue
            x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
            bw, bh = x2 - x1, y2 - y1
            b = dict(area=(bw * bh) / (w * h), x=int(x1), y=int(y1),
                     bw=int(bw), bh=int(bh), cx=x1 + bw / 2, cy=y1 + bh / 2,
                     aspect=bh / max(bw, 1.0), conf=float(box.conf[0]))
            if b['area'] < MIN_AREA_FRAC:
                continue
            if base == 'DrumRed':
                red_drums.append(b)
            elif base not in feats or b['conf'] > feats[base].get('conf', 0):
                feats[base] = b
        for i, b in enumerate(sorted(red_drums, key=lambda d: d['cx'])[:3], 1):
            feats[f'DrumRed{i}'] = b       # numbered port->stbd, like the HSV path
        return feats

    # ---------------- truth plumbing ----------------
    def _discover_truth_topics(self):
        for name, types in self.get_topic_names_and_types():
            if name.startswith('/truth/rel/') and name not in self._truth_subs:
                lm = name.rsplit('/', 1)[-1]
                self._truth_subs[name] = self.create_subscription(
                    PointStamped, name,
                    lambda m, lm=lm: self._on_truth(lm, m), 10)

    def _on_truth(self, lm_name, msg):
        self._truth[lm_name] = (time.time(),
                                np.array([msg.point.x, msg.point.y, msg.point.z]))

    def _truth_for(self, det_name, est_xyz):
        """Truth body-FRD position for a detected feature: explicit map first,
        exact name second, else nearest fresh truth landmark of the same class."""
        max_age = self.get_parameter('truth_max_age').value
        now = time.time()
        fresh = {n: v for n, (tw, v) in self._truth.items() if now - tw <= max_age}
        cand = self._truth_map.get(det_name, det_name)
        if cand in fresh:
            return cand, fresh[cand]
        tok = _class_token(det_name)
        best, bname = None, None
        for n, v in fresh.items():
            if _class_token(n) != tok and not n.startswith(tok):
                continue
            d = float(np.linalg.norm(v - est_xyz))
            if best is None or d < best:
                best, bname = d, n
        return (bname, fresh[bname]) if bname else (None, None)

    def _heartbeat(self):
        if self._last_frame_wall is None:
            self.get_logger().warn('no images yet — sim topic is '
                                   '/sauvc_auv/camera_front/image_color (set image_topic)')
        elif time.time() - self._last_frame_wall > 5.0:
            self.get_logger().warn('image stream STALLED (>5 s)')

    # ---------------- blob extraction (HSV backend) ----------------
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
        feats = {}
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
                continue
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
        elev = -((b['cy'] / h) - 0.5) * vfov
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
        if self._yolo is not None:
            feats = self._classify_yolo(bgr, w, h)
        else:
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
            self._print_table(poses)

        self.pub_dbg.publish(self.bridge.cv2_to_imgmsg(dbg, encoding='bgr8'))
        if self.get_parameter('show_detections').value:
            cv2.imshow('gate_detector', dbg)
            cv2.waitKey(1)

    def _print_table(self, poses):
        print('\n─ ESTIMATED feature poses in body FRD (x fwd, y stbd, z down)'
              ' vs GROUND TRUTH ─ bearing +stbd ─')
        print(f"  {'feature':<16}{'x':>8}{'y':>8}{'z':>8}"
              f"{'range':>8}{'brg°':>8}{'elev°':>8}"
              f" │{'dx':>7}{'dy':>7}{'dz':>7}{'dR':>7}{'dbrg°':>7}{'delev°':>7}"
              f"  truth")
        for name in sorted(poses):
            pz = poses[name]
            est = np.array([pz['x'], pz['y'], pz['z']])
            row = (f"  {name:<16}{pz['x']:>+8.2f}{pz['y']:>+8.2f}{pz['z']:>+8.2f}"
                   f"{pz['range']:>8.2f}{math.degrees(pz['brg']):>+8.1f}"
                   f"{-math.degrees(pz['elev']):>+8.1f}")
            tname, txyz = self._truth_for(name, est)
            if txyz is None:
                row += (f" │{'--':>7}{'--':>7}{'--':>7}{'--':>7}{'--':>7}{'--':>7}"
                        f"  (run landmark_truth_node)")
            else:
                d = est - txyz
                t_rng = float(np.linalg.norm(txyz))
                t_brg = math.degrees(math.atan2(txyz[1], txyz[0]))
                t_elv = math.degrees(math.atan2(
                    txyz[2], math.hypot(txyz[0], txyz[1])))
                # detector elev is +up; truth table prints +down — compare in +down
                e_elv = -math.degrees(pz['elev'])
                row += (f" │{d[0]:>+7.2f}{d[1]:>+7.2f}{d[2]:>+7.2f}"
                        f"{pz['range'] - t_rng:>+7.2f}"
                        f"{math.degrees(pz['brg']) - t_brg:>+7.1f}"
                        f"{e_elv - t_elv:>+7.1f}  {tname}")
            print(row)


def main():
    rclpy.init()
    try:
        rclpy.spin(GateDetectorNode())
    except KeyboardInterrupt:
        pass
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
