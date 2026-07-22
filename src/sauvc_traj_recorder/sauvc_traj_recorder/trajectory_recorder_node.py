#!/usr/bin/env python3
"""trajectory_recorder_node — logs camera + IMU + pressure/depth + ground-truth odometry
to disk, for building VIO/visual-SLAM evaluation datasets.

All topic names are PARAMETERS, not hardcoded — this node doesn't know whether it's
talking to sim or hardware. That knowledge lives in the two launch files
(record_trajectory_sim.launch.py / record_trajectory_hw.launch.py), which fill in the
right topic names for each platform. Leave a topic parameter empty to skip that stream
entirely (e.g. no odom_topic on hardware, since ground truth doesn't exist there).

Sub (all optional except output_dir):
    front_image_topic, front_info_topic   sensor_msgs/Image, sensor_msgs/CameraInfo
    down_image_topic,  down_info_topic    sensor_msgs/Image, sensor_msgs/CameraInfo
    imu_topic                             sensor_msgs/Imu
    pressure_topic                        sensor_msgs/FluidPressure   (sim: raw pressure)
    depth_topic                           geometry_msgs/PoseWithCovarianceStamped (hw: /depth)
    altitude_topic                        std_msgs/Float32                        (hw: /altitude)
    odom_topic                            nav_msgs/Odometry           (sim-only ground truth)

OUTPUT LAYOUT (output_dir is the per-trajectory folder; the launch file resolves it):
    camera_front/                 000000_<t>.png, 000001_<t>.png, ...  (per-frame, lossless by default)
    camera_down/
    camera_front.mp4              quick-look PREVIEW ONLY — see "ON THE VIDEO FILES" below
    camera_down.mp4
    camera_front_index.csv        idx, sec, nanosec, t, filename
    camera_down_index.csv
    camera_front_info.yaml        intrinsics snapshot (captured once, cameras are static)
    camera_down_info.yaml
    imu.csv                       idx, sec, nanosec, t, qx,qy,qz,qw, wx,wy,wz, ax,ay,az
    pressure.csv                  idx, sec, nanosec, t, fluid_pressure, variance      [sim]
    depth.csv                     idx, sec, nanosec, t, depth_m, depth_var            [hw]
    altitude.csv                  idx, sec, nanosec, t, altitude_m                    [hw]
    odometry.csv                  idx, sec, nanosec, t, x,y,z, qx,qy,qz,qw, vx,vy,vz,wx,wy,wz  [sim]
    odometry_anchored.csv         same, x/y/z shifted so the trajectory starts at (0,0,0) [sim]
    meta.yaml                     counts, rates, and a per-stream sync/gap report

ON TIMESTAMPS AND "SYNC"
-------------------------
Every row uses the message's own header.stamp, never local arrival time. Stonefish (and,
as long as nothing sets use_sim_time, the real driver stack) stamps every sensor with the
same wall clock, so every stream recorded here already shares one time base — that's what
makes offline association possible at all.

What this node does NOT do is force streams into lockstep (e.g. via an
ApproximateTimeSynchronizer). Cameras, IMU and pressure/depth arrive at different,
non-integer rates on purpose — that's what a VIO pipeline actually wants: raw per-sensor
timing, not something resampled onto a shared grid and quietly missing frames that didn't
line up. Pair frames with the nearest IMU/odometry sample OFFLINE instead, e.g.:

    import pandas as pd
    imu = pd.read_csv('imu.csv').sort_values('t')
    frames = pd.read_csv('camera_down_index.csv').sort_values('t')
    merged = pd.merge_asof(frames, imu, on='t', direction='nearest')

To catch the failure modes that WOULD break a downstream VIO run — dropped messages,
out-of-order delivery, a sensor that started late or stopped early — every stream is
tracked with a small StreamStats accumulator (count, time span, average rate, largest
inter-message gap, non-monotonic-timestamp count) and warned on live if a timestamp goes
backwards. The full report is written to meta.yaml at shutdown, including the overlap
window across all active streams, so you can tell before you feed this into ORB-SLAM3 /
VINS / OpenVINS / etc. whether the take is actually usable.

ON THE VIDEO FILES
--------------------
camera_front.mp4 / camera_down.mp4 are a QUICK-LOOK PREVIEW ONLY. A video container needs
one fixed fps, but the camera's real capture rate is irregular (rendering-cost dependent
in sim; whatever the driver delivers on hardware) — so playback speed will drift from real
time. Every frame in the video is still the same frame saved to camera_*/, in the same
order, so it's fine for scrubbing through a take to see what happened. For anything
quantitative (VIO, timing, frame-to-IMU association) use the per-frame images plus their
index CSV, not the video.

ON THE SPAWN-POSITION OFFSET
------------------------------
odometry.csv is absolute world position — it starts wherever the vehicle spawned
(start_position in the .scn, e.g. "-12.1 0 0.3"), not at (0,0,0). Any estimator you
compare it against starts at ITS OWN origin, so a raw comparison carries a fixed offset
that has nothing to do with estimator error. odometry.csv is deliberately left absolute
(that's what "ground truth" should mean); odometry_anchored.csv is a translation-only
convenience copy (matches flow_eval_node's own dead-reckoning anchor) for this project's
AHRS-referenced estimators. A general VIO/SLAM pipeline's frame is usually ROTATED
relative to this one too (and unscaled, if monocular) — see trajectory_tools/ for the
Umeyama SE(3)/Sim(3) alignment that case actually needs.
"""

import csv
import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, CameraInfo, Imu, FluidPressure
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry

from cv_bridge import CvBridge
import cv2


CAMS = ('camera_front', 'camera_down')


class StreamStats:
    """Tracks count / time span / rate / largest gap / ordering for one recorded stream."""

    def __init__(self, name, logger):
        self.name = name
        self.logger = logger
        self.count = 0
        self.first_t = None
        self.last_t = None
        self.max_gap = 0.0
        self.nonmonotonic = 0

    def update(self, t):
        if self.first_t is None:
            self.first_t = t
        elif self.last_t is not None:
            dt = t - self.last_t
            if dt < 0:
                self.nonmonotonic += 1
                self.logger.warn(
                    f'{self.name}: out-of-order timestamp (\u0394t={dt:.4f}s) - '
                    'downstream VIO/SLAM will mishandle this if not sorted first')
            elif dt > self.max_gap:
                self.max_gap = dt
        self.last_t = t
        self.count += 1

    def summary(self):
        span = (self.last_t - self.first_t) if self.count > 1 else 0.0
        hz = (self.count - 1) / span if span > 0 else 0.0
        return {
            'count': self.count,
            'first_t': round(self.first_t, 6) if self.first_t is not None else None,
            'last_t': round(self.last_t, 6) if self.last_t is not None else None,
            'span_s': round(span, 3),
            'avg_hz': round(hz, 2),
            'max_gap_s': round(self.max_gap, 3),
            'nonmonotonic': self.nonmonotonic,
        }


class TrajectoryRecorderNode(Node):
    def __init__(self):
        super().__init__('trajectory_recorder_node')
        p = self.declare_parameter
        p('output_dir', '')            # REQUIRED - set by the launch file per trajectory
        p('front_image_topic', '')
        p('front_info_topic', '')
        p('down_image_topic', '')
        p('down_info_topic', '')
        p('imu_topic', '')
        p('pressure_topic', '')
        p('depth_topic', '')
        p('altitude_topic', '')
        p('odom_topic', '')
        p('image_format', 'png')       # 'png' (lossless) or 'jpg' (smaller, lossy)
        p('jpeg_quality', 95)
        p('video_fps', 10.0)           # nominal container fps - see "ON THE VIDEO FILES"
        p('status_period', 5.0)        # s between "still recording" log lines
        g = lambda n: self.get_parameter(n).value

        self.output_dir = g('output_dir')
        if not self.output_dir:
            self.get_logger().fatal(
                "output_dir parameter is empty - run this via "
                "record_trajectory_sim.launch.py or record_trajectory_hw.launch.py, "
                "which resolve a per-trajectory folder.")
            raise SystemExit(1)
        os.makedirs(self.output_dir, exist_ok=True)

        self.image_format = g('image_format').lower()
        if self.image_format not in ('png', 'jpg', 'jpeg'):
            self.get_logger().warn(
                f"unknown image_format '{self.image_format}', defaulting to png")
            self.image_format = 'png'
        self.jpeg_quality = int(g('jpeg_quality'))
        self.video_fps = float(g('video_fps'))

        self.bridge = CvBridge()
        self.stats = {}       # name -> StreamStats
        self.counts = {}      # name -> int, mirrors stats[name].count for quick access
        self._warned_convert = set()

        # --- cameras (each optional) ---
        self.cam_dirs, self.cam_files, self.cam_writers = {}, {}, {}
        self.video_writers = {}
        self.info_saved = {}
        cam_topics = {
            'camera_front': (g('front_image_topic'), g('front_info_topic')),
            'camera_down': (g('down_image_topic'), g('down_info_topic')),
        }
        for cam, (image_topic, info_topic) in cam_topics.items():
            if not image_topic:
                continue
            d = os.path.join(self.output_dir, cam)
            os.makedirs(d, exist_ok=True)
            self.cam_dirs[cam] = d
            f = open(os.path.join(self.output_dir, f'{cam}_index.csv'), 'w', newline='')
            w = csv.writer(f)
            w.writerow(['idx', 'sec', 'nanosec', 't', 'filename'])
            self.cam_files[cam], self.cam_writers[cam] = f, w
            self.video_writers[cam] = None   # lazily opened on first frame
            self.info_saved[cam] = False
            self.counts[cam] = 0
            self.stats[cam] = StreamStats(cam, self.get_logger())
            self.create_subscription(
                Image, image_topic, (lambda msg, c=cam: self.on_image(msg, c)),
                qos_profile_sensor_data)
            if info_topic:
                self.create_subscription(
                    CameraInfo, info_topic, (lambda msg, c=cam: self.on_info(msg, c)),
                    qos_profile_sensor_data)

        # --- imu (optional) ---
        self.imu_writer = self.imu_file = None
        imu_topic = g('imu_topic')
        if imu_topic:
            self.imu_file = open(os.path.join(self.output_dir, 'imu.csv'), 'w', newline='')
            self.imu_writer = csv.writer(self.imu_file)
            self.imu_writer.writerow(
                ['idx', 'sec', 'nanosec', 't', 'qx', 'qy', 'qz', 'qw',
                 'wx', 'wy', 'wz', 'ax', 'ay', 'az'])
            self.counts['imu'] = 0
            self.stats['imu'] = StreamStats('imu', self.get_logger())
            self.create_subscription(Imu, imu_topic, self.on_imu, qos_profile_sensor_data)

        # --- pressure (sim: raw FluidPressure) ---
        self.pressure_writer = self.pressure_file = None
        pressure_topic = g('pressure_topic')
        if pressure_topic:
            self.pressure_file = open(os.path.join(self.output_dir, 'pressure.csv'), 'w', newline='')
            self.pressure_writer = csv.writer(self.pressure_file)
            self.pressure_writer.writerow(['idx', 'sec', 'nanosec', 't', 'fluid_pressure', 'variance'])
            self.counts['pressure'] = 0
            self.stats['pressure'] = StreamStats('pressure', self.get_logger())
            self.create_subscription(
                FluidPressure, pressure_topic, self.on_pressure, qos_profile_sensor_data)

        # --- depth / altitude (hardware: already-derived values) ---
        self.depth_writer = self.depth_file = None
        depth_topic = g('depth_topic')
        if depth_topic:
            self.depth_file = open(os.path.join(self.output_dir, 'depth.csv'), 'w', newline='')
            self.depth_writer = csv.writer(self.depth_file)
            self.depth_writer.writerow(['idx', 'sec', 'nanosec', 't', 'depth_m', 'depth_var'])
            self.counts['depth'] = 0
            self.stats['depth'] = StreamStats('depth', self.get_logger())
            self.create_subscription(
                PoseWithCovarianceStamped, depth_topic, self.on_depth, qos_profile_sensor_data)

        self.altitude_writer = self.altitude_file = None
        altitude_topic = g('altitude_topic')
        if altitude_topic:
            self.altitude_file = open(os.path.join(self.output_dir, 'altitude.csv'), 'w', newline='')
            self.altitude_writer = csv.writer(self.altitude_file)
            self.altitude_writer.writerow(['idx', 'sec', 'nanosec', 't', 'altitude_m'])
            self.counts['altitude'] = 0
            self.stats['altitude'] = StreamStats('altitude', self.get_logger())
            # Float32 has no header/stamp -> timestamp with node clock at arrival.
            self.create_subscription(
                Float32, altitude_topic, self.on_altitude, qos_profile_sensor_data)

        # --- ground truth odometry (sim only) ---
        self.odom_writer = self.odom_file = None
        self.odom_anchor_writer = self.odom_anchor_file = None
        self._odom_anchor = None   # (x0, y0, z0) of the first sample, set on first on_odom
        odom_topic = g('odom_topic')
        if odom_topic:
            self.odom_file = open(os.path.join(self.output_dir, 'odometry.csv'), 'w', newline='')
            self.odom_writer = csv.writer(self.odom_file)
            self.odom_writer.writerow([
                'idx', 'sec', 'nanosec', 't', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw',
                'vx', 'vy', 'vz', 'wx', 'wy', 'wz'])
            # Convenience copy, position-only anchored to the first sample (x,y,z all
            # shifted so the trajectory starts at 0,0,0 — orientation/velocity untouched).
            # This is a plain TRANSLATION offset, nothing more — it matches exactly what
            # flow_eval_node._anchor does for this project's own AHRS-referenced
            # dead-reckoning estimators (EKF/GTSAM), where axes already match ground
            # truth and only the origin differs. It is NOT a substitute for proper SE(3)/
            # Sim(3) alignment when scoring a general VIO/SLAM pipeline, whose frame is
            # usually rotated (and, for monocular, unscaled) relative to this one — for
            # that, see trajectory_tools/align_and_evaluate.py.
            self.odom_anchor_file = open(
                os.path.join(self.output_dir, 'odometry_anchored.csv'), 'w', newline='')
            self.odom_anchor_writer = csv.writer(self.odom_anchor_file)
            self.odom_anchor_writer.writerow([
                'idx', 'sec', 'nanosec', 't', 'x', 'y', 'z', 'qx', 'qy', 'qz', 'qw',
                'vx', 'vy', 'vz', 'wx', 'wy', 'wz'])
            self.counts['odometry'] = 0
            self.stats['odometry'] = StreamStats('odometry', self.get_logger())
            self.create_subscription(
                Odometry, odom_topic, self.on_odom, qos_profile_sensor_data)

        if not self.cam_dirs:
            self.get_logger().warn(
                'no camera topics configured (front_image_topic and down_image_topic '
                'are both empty) - recording other streams only, if any.')

        self.start_wall = time.time()
        self.start_iso = datetime.now().isoformat()
        self.create_timer(float(g('status_period')), self.report)

        active = [k for k in ('camera_front', 'camera_down', 'imu', 'pressure',
                              'depth', 'altitude', 'odometry') if k in self.counts]
        self.get_logger().info(
            f"trajectory_recorder up -> {self.output_dir}\n"
            f"  recording: {', '.join(active) if active else '(nothing configured!)'}\n"
            f"  image_format={self.image_format}  video_fps={self.video_fps} (preview only)\n"
            f"  Ctrl+C to stop this trajectory.")

    # ---- camera ----
    def _get_video_writer(self, cam, frame):
        vw = self.video_writers.get(cam)
        if vw is not None or cam not in self.video_writers:
            return vw
        h, w = frame.shape[:2]
        path = os.path.join(self.output_dir, f'{cam}.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        vw = cv2.VideoWriter(path, fourcc, self.video_fps, (w, h))
        if not vw.isOpened():
            self.get_logger().warn(
                f'{cam}: could not open video writer for {path} (codec/container '
                'issue) - continuing with per-frame images only, no video.')
            vw = False   # sentinel: tried once, don't retry every frame
        self.video_writers[cam] = vw
        return vw

    def on_image(self, msg, cam):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            if cam not in self._warned_convert:
                self._warned_convert.add(cam)
                self.get_logger().error(f'{cam}: image conversion failed ({e}); dropping frames')
            return

        idx = self.counts[cam]
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.stats[cam].update(t)

        fname = f'{idx:06d}_{t:.6f}.{self.image_format}'
        path = os.path.join(self.cam_dirs[cam], fname)
        if self.image_format in ('jpg', 'jpeg'):
            cv2.imwrite(path, frame, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        else:
            cv2.imwrite(path, frame)

        vw = self._get_video_writer(cam, frame)
        if vw:
            vw.write(frame)

        self.cam_writers[cam].writerow(
            [idx, msg.header.stamp.sec, msg.header.stamp.nanosec, f'{t:.6f}', fname])
        self.cam_files[cam].flush()
        self.counts[cam] += 1

    def on_info(self, msg, cam):
        # Cameras are rigidly mounted and static; one snapshot suffices.
        if self.info_saved[cam]:
            return
        self.info_saved[cam] = True
        path = os.path.join(self.output_dir, f'{cam}_info.yaml')
        with open(path, 'w') as f:
            f.write(f'# camera_info snapshot for {cam}, captured once at recording start\n')
            f.write(f'width: {msg.width}\n')
            f.write(f'height: {msg.height}\n')
            f.write(f'distortion_model: {msg.distortion_model}\n')
            f.write('k: [' + ', '.join(f'{v:.6f}' for v in msg.k) + ']\n')
            f.write('d: [' + ', '.join(f'{v:.6f}' for v in msg.d) + ']\n')
        self.get_logger().info(f'{cam}: saved intrinsics -> {path}')

    # ---- imu ----
    def on_imu(self, msg):
        idx = self.counts['imu']
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.stats['imu'].update(t)
        q, w, a = msg.orientation, msg.angular_velocity, msg.linear_acceleration
        self.imu_writer.writerow([
            idx, msg.header.stamp.sec, msg.header.stamp.nanosec, f'{t:.6f}',
            q.x, q.y, q.z, q.w, w.x, w.y, w.z, a.x, a.y, a.z])
        self.imu_file.flush()
        self.counts['imu'] += 1

    # ---- pressure / depth / altitude ----
    def on_pressure(self, msg):
        idx = self.counts['pressure']
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.stats['pressure'].update(t)
        self.pressure_writer.writerow(
            [idx, msg.header.stamp.sec, msg.header.stamp.nanosec, f'{t:.6f}',
             msg.fluid_pressure, msg.variance])
        self.pressure_file.flush()
        self.counts['pressure'] += 1

    def on_depth(self, msg):
        idx = self.counts['depth']
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.stats['depth'].update(t)
        depth_m = -msg.pose.pose.position.z   # /depth convention: position.z = -depth
        depth_var = msg.pose.covariance[14]
        self.depth_writer.writerow(
            [idx, msg.header.stamp.sec, msg.header.stamp.nanosec, f'{t:.6f}', depth_m, depth_var])
        self.depth_file.flush()
        self.counts['depth'] += 1

    def on_altitude(self, msg):
        # std_msgs/Float32 carries no header - stamp with node clock at arrival.
        idx = self.counts['altitude']
        now = self.get_clock().now().to_msg()
        t = now.sec + now.nanosec * 1e-9
        self.stats['altitude'].update(t)
        self.altitude_writer.writerow([idx, now.sec, now.nanosec, f'{t:.6f}', msg.data])
        self.altitude_file.flush()
        self.counts['altitude'] += 1

    # ---- ground truth odometry (sim only) ----
    def on_odom(self, msg):
        idx = self.counts['odometry']
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.stats['odometry'].update(t)
        pos, q = msg.pose.pose.position, msg.pose.pose.orientation
        v, w = msg.twist.twist.linear, msg.twist.twist.angular
        self.odom_writer.writerow([
            idx, msg.header.stamp.sec, msg.header.stamp.nanosec, f'{t:.6f}',
            pos.x, pos.y, pos.z, q.x, q.y, q.z, q.w, v.x, v.y, v.z, w.x, w.y, w.z])
        self.odom_file.flush()

        if self._odom_anchor is None:
            self._odom_anchor = (pos.x, pos.y, pos.z)
        x0, y0, z0 = self._odom_anchor
        self.odom_anchor_writer.writerow([
            idx, msg.header.stamp.sec, msg.header.stamp.nanosec, f'{t:.6f}',
            pos.x - x0, pos.y - y0, pos.z - z0, q.x, q.y, q.z, q.w,
            v.x, v.y, v.z, w.x, w.y, w.z])
        self.odom_anchor_file.flush()

        self.counts['odometry'] += 1

    # ---- status / shutdown ----
    def report(self):
        elapsed = max(time.time() - self.start_wall, 1e-6)
        parts = [f"{k}={self.counts[k]} ({self.counts[k] / elapsed:.1f} Hz)"
                 for k in self.counts]
        self.get_logger().info(
            f"[{os.path.basename(self.output_dir)}] " + '  '.join(parts) +
            f"  elapsed={elapsed:.0f}s")

    def finalize(self):
        for f in self.cam_files.values():
            f.close()
        for vw in self.video_writers.values():
            if vw:
                vw.release()
        for f in (self.imu_file, self.pressure_file, self.depth_file,
                  self.altitude_file, self.odom_file, self.odom_anchor_file):
            if f:
                f.close()

        elapsed = max(time.time() - self.start_wall, 1e-6)
        summaries = {name: s.summary() for name, s in self.stats.items()}

        firsts = [s['first_t'] for s in summaries.values() if s['first_t'] is not None]
        lasts = [s['last_t'] for s in summaries.values() if s['last_t'] is not None]
        overlap_start = max(firsts) if firsts else None
        overlap_end = min(lasts) if lasts else None
        overlap_dur = (overlap_end - overlap_start) if (overlap_start is not None
                                                         and overlap_end is not None) else 0.0

        meta_path = os.path.join(self.output_dir, 'meta.yaml')
        with open(meta_path, 'w') as f:
            f.write(f"trajectory: {os.path.basename(self.output_dir)}\n")
            f.write(f"start_time: {self.start_iso}\n")
            f.write(f"end_time: {datetime.now().isoformat()}\n")
            f.write(f"duration_sec: {elapsed:.2f}\n")
            f.write("streams:\n")
            for name, s in summaries.items():
                f.write(f"  {name}:\n")
                for k, v in s.items():
                    f.write(f"    {k}: {v}\n")
            f.write("sync:\n")
            f.write(f"  overlap_start_t: {round(overlap_start, 6) if overlap_start is not None else None}\n")
            f.write(f"  overlap_end_t: {round(overlap_end, 6) if overlap_end is not None else None}\n")
            f.write(f"  overlap_duration_sec: {round(overlap_dur, 3)}\n")
            f.write("  note: overlap window is where ALL active streams have data; trim\n")
            f.write("    to this window before feeding a VIO/SLAM pipeline. See each\n")
            f.write("    stream's max_gap_s / nonmonotonic above for drop/ordering issues.\n")
            f.write("frame_convention: "
                    "ground truth (if recorded) is Stonefish-native NED world / FRD body, "
                    "unconverted; imu.csv is whatever imu_topic publishes (ENU/FLU on both "
                    "sim's /imu/data shim and real hardware)\n")
            f.write("video_note: camera_*.mp4 is a fixed-fps preview only "
                    f"(nominal {self.video_fps} fps) - real per-frame timing is in "
                    "camera_*_index.csv, not the video.\n")
            f.write("bag: bag/  (present if record_bag:=true was used; "
                    "same topics, ros2 bag play-able)\n")
            if self.odom_writer is not None:
                f.write("ground_truth_files:\n")
                f.write("  odometry.csv: absolute world position, exactly as the sensor "
                        "reports it (spawn offset included) — this is the reference; "
                        "never modified.\n")
                f.write("  odometry_anchored.csv: same data, x/y/z shifted so the "
                        "trajectory starts at (0,0,0) — a plain translation, matching "
                        "flow_eval_node's dead-reckoning anchor. Fine for this project's "
                        "AHRS-referenced EKF/GTSAM estimators; NOT sufficient for a "
                        "general VIO/SLAM pipeline (arbitrary rotation, unknown scale "
                        "if monocular) — use trajectory_tools/align_and_evaluate.py "
                        "(Umeyama SE(3)/Sim(3), fit over the whole trajectory) for that.\n")

        # flag anything that looks off, right when it's actionable
        for name, s in summaries.items():
            if s['nonmonotonic']:
                self.get_logger().warn(
                    f'{name}: {s["nonmonotonic"]} out-of-order timestamp(s) recorded')
            if s['count'] > 1 and s['max_gap_s'] > 5.0 / max(s['avg_hz'], 0.1):
                self.get_logger().warn(
                    f'{name}: largest gap between messages was {s["max_gap_s"]:.2f}s '
                    f'(avg rate {s["avg_hz"]:.1f} Hz) - check for a dropout mid-take')

        self.get_logger().info(
            f"trajectory saved: {self.output_dir}  ({elapsed:.0f}s, "
            f"overlap window {overlap_dur:.1f}s) -> {meta_path}")


def main():
    rclpy.init()
    node = TrajectoryRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.finalize()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
