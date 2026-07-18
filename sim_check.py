#!/usr/bin/env python3
"""sim_check.py — one-shot health check for the Stonefish sim. No build required.

    cd ~/Robotics_Job/sauvc_ws
    python3 sim_check.py                 # 10 s measurement window
    python3 sim_check.py --secs 20

Run it with the simulator up. It answers, in one go, every question that currently
blocks the localization translation:

  1. REAL-TIME FACTOR -- the big one. Each sensor's rate="..." in my_auv.scn is in
     SIMULATION time, but stonefish_ros2 stamps every message with the WALL CLOCK
     (ROS2Interface.cpp uses nh_->get_clock()->now() in all 20 publishers and
     s.getTimestamp() in none). So observed wall-clock rate / declared rate == RTF.

     This matters because flow_velocity_node derives dt from image header stamps. At
     real-time factor R it reports R * v_true, while gyro rates, depth and a DVL are
     NOT scaled. R != 1 does not make the sim merely slow -- it makes it KINEMATICALLY
     INCONSISTENT, and flow_scorer would report a scale error of exactly R for an
     algorithm that is working perfectly.

  2. Whether all topics degrade TOGETHER. Stonefish steps every sensor from one
     simulation loop, so a genuine RTF shortfall hits all of them proportionally. If
     only the cameras are down, that is per-sensor throttling, not RTF -- a different
     problem with a different fix.

  3. /clock -- expected ABSENT, which forces use_sim_time:=false everywhere.

  4. Camera intrinsics from camera_info, to confirm what goes in flow_sim.yaml.

  5. Pressure sign/scale, to confirm gauge-vs-absolute and rho*g.

Nothing here is fused or published. It is a read-only diagnostic.
"""
import argparse
import math
import sys
import time
from collections import defaultdict

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, FluidPressure, Image, CameraInfo
from nav_msgs.msg import Odometry

# name -> (topic, type, declared rate from my_auv.scn)
WATCH = [
    ('imu',          '/sauvc_auv/imu',                      Imu,           100.0),
    ('pressure',     '/sauvc_auv/pressure',                 FluidPressure,  20.0),
    ('odometry',     '/sauvc_auv/odometry',                 Odometry,       30.0),
    ('camera_down',  '/sauvc_auv/camera_down/image_color',  Image,          30.0),
    ('camera_front', '/sauvc_auv/camera_front/image_color', Image,          30.0),
]

RHO, G = 1000.0, 9.81   # scene <water density="1000.0"/>; Stonefish g


class SimCheck(Node):
    def __init__(self, secs):
        super().__init__('sim_check')
        self.secs = secs
        self.counts = defaultdict(int)
        self.first_stamp = {}
        self.last_stamp = {}
        self.first_wall = {}
        self.last_wall = {}
        self.cam_info = {}
        self.last_press = None
        self.last_imu = None
        self.last_odom = None
        self.odom_pairs = []
        self._prev_odom = None

        for name, topic, typ, _ in WATCH:
            self.create_subscription(
                typ, topic, lambda m, n=name: self.on_msg(m, n), qos_profile_sensor_data)
        for cam in ('camera_down', 'camera_front'):
            self.create_subscription(
                CameraInfo, f'/sauvc_auv/{cam}/camera_info',
                lambda m, c=cam: self.cam_info.setdefault(c, m), qos_profile_sensor_data)
        self.t0 = time.time()

    def on_msg(self, msg, name):
        now = time.time()
        self.counts[name] += 1
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if name not in self.first_wall:
            self.first_wall[name] = now
            self.first_stamp[name] = stamp
        self.last_wall[name] = now
        self.last_stamp[name] = stamp

        if name == 'pressure':
            self.last_press = msg
        elif name == 'imu':
            self.last_imu = msg
        elif name == 'odometry':
            self.last_odom = msg
            p = msg.pose.pose.position
            v = msg.twist.twist.linear
            speed = math.sqrt(v.x**2 + v.y**2 + v.z**2)
            if self._prev_odom is not None:
                (pt, px, py, pz) = self._prev_odom
                dt = stamp - pt
                if dt > 1e-6 and speed > 0.05:
                    apparent = math.dist((px, py, pz), (p.x, p.y, p.z)) / dt
                    self.odom_pairs.append(apparent / speed)
            self._prev_odom = (stamp, p.x, p.y, p.z)


def hr(t=''):
    print(f'\n{"─"*72}\n{t}' if t else '─'*72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--secs', type=float, default=10.0)
    args = ap.parse_args()

    rclpy.init()
    node = SimCheck(args.secs)

    print(f'measuring for {args.secs:.0f} s — leave the sim running…')
    end = time.time() + args.secs
    while time.time() < end and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.05)

    topic_names = [t for t, _ in node.get_topic_names_and_types()]

    hr('1. REAL-TIME FACTOR   (observed wall rate ÷ declared sim rate)')
    print(f'{"sensor":<14}{"declared":>10}{"observed":>11}{"RTF":>8}')
    rtfs = {}
    for name, topic, _, declared in WATCH:
        n = node.counts[name]
        if n < 2:
            print(f'{name:<14}{declared:>10.1f}{"NO DATA":>11}{"—":>8}')
            continue
        span = node.last_wall[name] - node.first_wall[name]
        obs = (n - 1) / span if span > 1e-6 else 0.0
        r = obs / declared
        rtfs[name] = r
        print(f'{name:<14}{declared:>10.1f}{obs:>11.2f}{r:>8.3f}')

    if node.odom_pairs:
        kin = sorted(node.odom_pairs)[len(node.odom_pairs)//2]
        print(f'\nkinematic cross-check (|Δpos|/dt ÷ |twist|): {kin:.3f}   '
              f'[{len(node.odom_pairs)} samples]')
    else:
        print('\nkinematic cross-check: n/a — vehicle stationary (this is expected '
              'if it is just floating). The rate method above is still valid.')

    hr('VERDICT')
    if not rtfs:
        print('✗ no data on any topic. Is the sim actually running?')
    else:
        vals = list(rtfs.values())
        lo, hi = min(vals), max(vals)
        spread = hi - lo
        mean = sum(vals) / len(vals)
        if spread > 0.15:
            worst = min(rtfs, key=rtfs.get)
            print(f'⚠ topics DISAGREE (spread {spread:.2f}; worst: {worst} at '
                  f'{rtfs[worst]:.3f}).')
            print('  Stonefish steps all sensors from one sim loop, so a genuine RTF')
            print('  shortfall hits everything proportionally. Disagreement means')
            print('  PER-SENSOR throttling, not RTF — likely the cameras are')
            print('  GPU-bound. Different problem, different fix.')
        elif abs(mean - 1.0) <= 0.05:
            print(f'✓ RTF ≈ {mean:.3f}. The sim is real-time and kinematically')
            print('  consistent. flow_velocity_node\'s wall-clock dt matches physics,')
            print('  so flow scale is trustworthy. Proceed.')
        else:
            print(f'✗ RTF ≈ {mean:.3f}, NOT 1.0.')
            print(f'  flow_velocity_node will report ~{mean:.2f}× true velocity while')
            print('  gyro/depth/DVL are unscaled. The EKF would fuse inconsistent')
            print('  kinematics, and flow_scorer would blame your calibration for a')
            print(f'  {(mean-1)*100:+.0f}% "scale error" that is really a timing artifact.')
            print('  Fix before Phase 3: drop camera resolution to 640×480 in')
            print('  my_auv.scn, or lower rendering_quality in the launch file.')

    hr('2. /clock')
    if '/clock' in topic_names:
        print('⚠ /clock EXISTS — unexpected. Re-examine the use_sim_time decision.')
    else:
        print('✓ absent, as expected. Stonefish stamps with the wall clock, so')
        print('  use_sim_time MUST stay false everywhere.')

    hr('3. CAMERA INTRINSICS   (paste into flow_sim.yaml)')
    for cam, m in node.cam_info.items():
        fx, fy, cx, cy = m.k[0], m.k[4], m.k[2], m.k[5]
        d = list(m.d)
        print(f'{cam}: {m.width}×{m.height}  fx={fx:.2f} fy={fy:.2f} '
              f'cx={cx:.1f} cy={cy:.1f}')
        print(f'{"":12}distortion={d}  '
              f'{"(zero — no calibration exists in sim)" if not any(d) else "⚠ NONZERO"}')
        if abs(fx - fy) > 1e-6:
            print(f'{"":12}⚠ fx != fy — unexpected; Stonefish derives fy from fx.')
    if not node.cam_info:
        print('no camera_info received.')

    hr('4. PRESSURE')
    if node.last_press:
        p, var = node.last_press.fluid_pressure, node.last_press.variance
        print(f'fluid_pressure = {p:.3f} Pa   variance = {var:.1f} Pa²  '
              f'(σ = {math.sqrt(var):.1f} Pa)')
        print(f'depth = P/(ρg) = {p/(RHO*G)*1000:+.2f} mm   using ρ={RHO} g={G} '
              f'(ρg={RHO*G:.1f})')
        print(f'σ_depth = {math.sqrt(var)/(RHO*G)*1000:.2f} mm  → '
              f'depth_var = {(math.sqrt(var)/(RHO*G))**2:.3e} m²')
        if p < 0:
            print('✓ negative → GAUGE pressure vs the free surface, no atmospheric')
            print('  term, not clamped above the waterline. p_ref=0 is correct.')
    if node.last_odom and node.last_press:
        # The one true empirical test of rho*g: ground-truth depth vs measured pressure.
        z_true = node.last_odom.pose.pose.position.z   # NED: +z is DOWN = depth
        z_sensor = z_true - 0.10                        # sensor sits 0.10 m above origin
        if abs(z_sensor) > 0.05:
            implied = node.last_press.fluid_pressure / z_sensor
            print(f'\nEMPIRICAL ρg CHECK (needs the vehicle submerged >5 cm):')
            print(f'  odometry z = {z_true:.4f} m (NED, +down) → sensor at '
                  f'{z_sensor:.4f} m')
            print(f'  implied ρg = P/depth = {implied:.1f}   vs configured '
                  f'{RHO*G:.1f}   ({(implied/(RHO*G)-1)*100:+.2f}%)')
        else:
            print('\nEMPIRICAL ρg CHECK: skipped — vehicle is at the surface.')
            print('  Dive it >5 cm and re-run to verify ρg against ground truth.')

    hr('5. WHAT IS MISSING FROM my_auv.scn')
    print('DVL          : ' + ('present' if '/sauvc_auv/dvl' in topic_names
                               else '✗ ABSENT — flow_scorer_node cannot run. See '
                                    'SCENE_CHANGES.md §2.'))
    if node.last_imu:
        yaw_var = node.last_imu.orientation_covariance[8]
        acc_var = node.last_imu.linear_acceleration_covariance[0]
        print(f'IMU yaw σ    : {math.sqrt(yaw_var):.2e} rad '
              f'({math.degrees(math.sqrt(yaw_var)):.5f}°)')
        if yaw_var < 1e-8:
            print('               ✗ near-perfect AHRS, and no yaw_drift in the scene →')
            print('                 sim yaw CANNOT drift → lane_heading_node has nothing')
            print('                 to correct and its tests pass vacuously. '
                  'SCENE_CHANGES.md §1.')
        print(f'accel cov[0] : {acc_var:.3e}' +
              ('  ✗ ZERO = "perfectly known" per the Imu spec (unknown is -1). '
               'Add <noise linear_acceleration="..."/>.' if acc_var == 0.0 else ''))
    print('USBL         : ' + ('present — NOTE: the real vehicle has no USBL. Keep it '
                               'inside /sauvc_auv/ and never fuse it.'
                               if '/sauvc_auv/usbl' in topic_names else 'absent'))
    hr()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
