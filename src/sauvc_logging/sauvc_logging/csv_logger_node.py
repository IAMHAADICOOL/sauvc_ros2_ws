#!/usr/bin/env python3
"""csv_logger_node — automatic CSV logging for every phase, so a pool session leaves a
data trail without anyone manually running `ros2 topic echo --csv` and timing a Ctrl-C.

What it does:
  - Subscribes to a configurable list of topics (this project's known set is in
    TOPIC_REGISTRY below — add an entry there if you introduce a new topic).
  - Writes ONE CSV per topic under a timestamped run directory, one row per message,
    with both the message's own header stamp and a clean set of extracted fields.
  - On shutdown (Ctrl-C) prints a per-column mean/std/variance summary for every numeric
    column across the whole run — this replaces the manual estimate_covariance.py step
    for a single live run. estimate_covariance.py is still useful for combining multiple
    saved CSVs after the fact, or re-analysing a slice of a longer run.

Files land in: <out_dir>/<run_name>_<YYYYmmdd_HHMMSS>/<topic_sanitized>.csv

Standalone usage:
  ros2 run sauvc_logging csv_logger_node --ros-args \
      -p topics:="['/depth','/altitude']" -p run_name:=phase1_depth

Normally you won't run it standalone — each phaseN launch file in sauvc_bringup already
includes this node with a phase-appropriate topic list and run_name, so logging happens
automatically whenever you launch a phase.
"""
import csv
import math
import os
import time
from datetime import datetime

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32, String
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from nav_msgs.msg import Odometry


def _stamp_to_sec(stamp):
    return stamp.sec + stamp.nanosec * 1e-9


def _imu_cols(msg):
    return dict(
        stamp=_stamp_to_sec(msg.header.stamp),
        qx=msg.orientation.x, qy=msg.orientation.y,
        qz=msg.orientation.z, qw=msg.orientation.w,
        wx=msg.angular_velocity.x, wy=msg.angular_velocity.y, wz=msg.angular_velocity.z,
        ax=msg.linear_acceleration.x, ay=msg.linear_acceleration.y,
        az=msg.linear_acceleration.z,
        orient_cov_yaw=msg.orientation_covariance[8],
        gyro_cov_wz=msg.angular_velocity_covariance[8],
    )


def _depth_cols(msg):
    return dict(stamp=_stamp_to_sec(msg.header.stamp),
                z=msg.pose.pose.position.z, z_var=msg.pose.covariance[14])


def _flow_cols(msg):
    return dict(stamp=_stamp_to_sec(msg.header.stamp),
                vx=msg.twist.twist.linear.x, vy=msg.twist.twist.linear.y,
                var_vx=msg.twist.covariance[0], var_vy=msg.twist.covariance[7])


def _float_cols(msg):
    return dict(stamp=time.time(), value=msg.data)


def _odom_cols(msg):
    p = msg.pose.pose.position
    v = msg.twist.twist.linear
    return dict(stamp=_stamp_to_sec(msg.header.stamp),
                x=p.x, y=p.y, z=p.z, vx=v.x, vy=v.y, vz=v.z)


def _string_cols(msg):
    # e.g. /vision/detections: "label,bearing_rad,elev_rad,area_frac" — kept as text,
    # excluded from the numeric mean/std summary automatically (see close_and_summarize).
    return dict(stamp=time.time(), data=msg.data)


TOPIC_REGISTRY = {
    '/imu/data':              (Imu, _imu_cols),
    '/imu/data_corrected':    (Imu, _imu_cols),
    '/depth':                  (PoseWithCovarianceStamped, _depth_cols),
    '/altitude':               (Float32, _float_cols),
    '/flow/twist':              (TwistWithCovarianceStamped, _flow_cols),
    '/heading/pool_relative':   (Float32, _float_cols),
    '/heading/line_meas':       (Float32, _float_cols),
    '/odometry/filtered':       (Odometry, _odom_cols),
    '/odometry/preint':         (Odometry, _odom_cols),
    '/vision/detections':       (String, _string_cols),
}


class CsvLoggerNode(Node):
    def __init__(self):
        super().__init__('csv_logger_node')
        self.declare_parameter('topics', list(TOPIC_REGISTRY.keys()))
        self.declare_parameter('out_dir', os.path.expanduser('~/sauvc_logs'))
        self.declare_parameter('run_name', 'run')

        topics = self.get_parameter('topics').value
        out_dir = os.path.expanduser(self.get_parameter('out_dir').value)
        run_name = self.get_parameter('run_name').value
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = os.path.join(out_dir, f'{run_name}_{ts}')
        os.makedirs(self.run_dir, exist_ok=True)

        self.files = {}
        self.writers = {}
        self.buffers = {}   # topic -> {col: [values]} for the shutdown summary
        self._subs = []     # keep references alive

        for topic in topics:
            if topic not in TOPIC_REGISTRY:
                self.get_logger().warn(
                    f'"{topic}" not in TOPIC_REGISTRY — add it to csv_logger_node.py '
                    f'if you want it logged. Skipping.')
                continue
            msg_type, cols_fn = TOPIC_REGISTRY[topic]
            path = os.path.join(self.run_dir, topic.strip('/').replace('/', '_') + '.csv')
            self.files[topic] = open(path, 'w', newline='')
            self.writers[topic] = None    # created lazily once we see the first message
            self.buffers[topic] = {}
            sub = self.create_subscription(
                msg_type, topic,
                (lambda msg, t=topic, fn=cols_fn: self._on_msg(t, fn, msg)), 20)
            self._subs.append(sub)

        self.get_logger().info(f'logging {len(self.files)} topic(s) to {self.run_dir}')
        self.get_logger().info('Ctrl-C to stop — prints a mean/std summary on exit')

    def _on_msg(self, topic, cols_fn, msg):
        row = cols_fn(msg)
        w = self.writers[topic]
        if w is None:
            w = csv.DictWriter(self.files[topic], fieldnames=list(row.keys()))
            w.writeheader()
            self.writers[topic] = w
            for k in row:
                self.buffers[topic][k] = []
        w.writerow(row)
        for k, v in row.items():
            if isinstance(v, (int, float)):
                self.buffers[topic][k].append(v)

    def close_and_summarize(self):
        for f in self.files.values():
            f.flush()
            f.close()
        print(f'\n--- csv_logger_node summary: {self.run_dir} ---')
        for topic, cols in self.buffers.items():
            n = max((len(v) for v in cols.values()), default=0)
            if n == 0:
                continue
            print(f'{topic}  (n={n})')
            for col, vals in cols.items():
                if len(vals) < 2:
                    continue
                mean = sum(vals) / len(vals)
                var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
                print(f'    {col:16s} mean={mean:+.5f}  std={math.sqrt(var):.5f}  '
                      f'variance={var:.7g}')
        print(f'--- CSVs saved under {self.run_dir} ---\n')


def main():
    rclpy.init()
    node = CsvLoggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close_and_summarize()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
