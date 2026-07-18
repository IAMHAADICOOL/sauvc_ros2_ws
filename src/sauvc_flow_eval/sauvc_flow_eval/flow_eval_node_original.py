#!/usr/bin/env python3
"""flow_eval_node — compares localization approaches against Stonefish ground truth.

Runs FIVE estimators off the same live sensor stream and publishes each as its own
nav_msgs/Odometry on /eval/*, all in ONE common frame (default NED) so PlotJuggler can
overlay them directly:

  /eval/ground_truth   Stonefish odometry (the reference)
  /eval/flow           flow + altitude (metric via pressure), integrated to position
  /eval/ekf            self-contained EKF fusing flow + depth
  /eval/pressure       depth only (world Z; x,y left at 0)
  /eval/gtsam          IMU preintegration, integrated to position   [only if gtsam present]

Also opens two OpenCV windows: the raw down-camera feed and the optical-flow overlay.

DESIGN (per request):
  * Landmark/gate localization is deliberately NOT included — this is dead-reckoning-style
    comparison only.
  * Each estimator lives in its own module under estimators/; this node only wires them.
  * No ArduSub/SITL anywhere.
  * Modular for the real vehicle: it subscribes to the SHIMMED topics (/imu/data, /depth,
    /altitude, /camera_down/image_raw) which are identical on the real AUV, plus the
    sim-only /sauvc_auv/odometry for ground truth (absent on hardware -> that estimator
    simply stays silent).

FRAMES: `compare_frame` (default 'ned'). Ground truth is native NED; the ENU-native
estimates are converted with the tested sauvc_sim_bridge.frames conversion, in ONE place
(eval_common), never ad-hoc. See eval_common for the full rationale.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu, Image, FluidPressure
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry

try:
    from cv_bridge import CvBridge
    import cv2
    _HAVE_CV = True
except Exception:
    _HAVE_CV = False

from sauvc_flow_eval.eval_common import (
    PositionIntegrator, depth_to_world_z, to_compare_frame_world, gt_world_to_compare)
from sauvc_flow_eval.estimators.flow_estimator import FlowEstimator
from sauvc_flow_eval.estimators.ekf_estimator import EkfEstimator
from sauvc_flow_eval.estimators.gtsam_estimator import GtsamEstimator

from sauvc_sim_bridge.frames import ned_frd_quat_to_enu_flu, flu_frd_to_ned_wxyz


def _enu_quat_to_ned_wxyz(x, y, z, w):
    return flu_frd_to_ned_wxyz(x, y, z, w)


def _yaw_from_quat_xyzw(x, y, z, w):
    return np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


class FlowEvalNode(Node):
    def __init__(self):
        super().__init__('flow_eval_node')
        p = self.declare_parameter
        p('compare_frame', 'ned')          # 'ned' (default, per request) | 'enu'
        p('robot_name', 'sauvc_auv')
        p('fx', 762.72); p('fy', 762.72); p('cx', 640.0); p('cy', 360.0)
        p('pool_depth', 1.6)               # for altitude if /altitude is unavailable
        p('show_windows', True)          # master: any OpenCV window at all
        p('show_optical_flow', True)     # the optical-flow overlay window specifically
        p('show_camera', True)           # the raw down-camera window specifically
        p('print_estimates', True)       # print all 5 x/y/z to the terminal
        p('print_rate', 5.0)             # Hz, terminal print throttle
        p('gravity', 9.81)

        g = lambda n: self.get_parameter(n).value
        self.frame = g('compare_frame')
        if self.frame not in ('ned', 'enu'):
            raise ValueError("compare_frame must be 'ned' or 'enu'")
        self.show = g('show_windows') and _HAVE_CV
        self.show_flow = self.show and g('show_optical_flow')
        self.show_cam = self.show and g('show_camera')
        self.print_est = g('print_estimates')
        self.print_period = 1.0 / max(g('print_rate'), 0.1)
        # latest published value per source, for the terminal table
        self._latest = {k: None for k in
                        ('ground_truth', 'flow', 'ekf', 'pressure', 'gtsam')}
        self._last_print = 0.0

        # --- estimators ---
        self.flow = FlowEstimator(g('fx'), g('fy'), g('cx'), g('cy'))
        self.ekf = EkfEstimator()
        self.gtsam = GtsamEstimator(gravity=g('gravity'), compare_frame=self.frame)
        self.flow_pos = PositionIntegrator()
        self.gtsam_pos = PositionIntegrator()

        if self.gtsam.available:
            self.get_logger().info('gtsam present -> /eval/gtsam active')
        else:
            self.get_logger().warn('gtsam NOT available -> /eval/gtsam disabled '
                                   '(other four estimators run normally)')

        # --- state caches ---
        self.bridge = CvBridge() if _HAVE_CV else None
        self.gyro_body = (0.0, 0.0, 0.0)
        self.accel_body = (0.0, 0.0, 0.0)
        self.last_quat_wxyz_ned = (1.0, 0.0, 0.0, 0.0)
        self.yaw = 0.0          # compare-frame yaw (for flow integration / EKF)
        self.yaw_ned = 0.0      # NED yaw (for rotating flow into the graph's NED world)
        self.gt_anchor = None   # ground-truth first pose (compare frame); all tracks start here
        self.depth = 0.0
        self.altitude = None
        self.prev_img_t = None

        # --- publishers ---
        self.pubs = {k: self.create_publisher(Odometry, f'/eval/{k}', 10)
                     for k in ('ground_truth', 'flow', 'ekf', 'pressure', 'gtsam')}

        # --- subscriptions ---
        robot = g('robot_name')
        self.create_subscription(Imu, '/imu/data', self.on_imu, qos_profile_sensor_data)
        self.create_subscription(FluidPressure, f'/{robot}/pressure',
                                 self.on_pressure, qos_profile_sensor_data)
        self.create_subscription(Float32, '/altitude', self.on_altitude, 10)
        self.create_subscription(Image, '/camera_down/image_raw',
                                 self.on_image, qos_profile_sensor_data)
        self.create_subscription(Odometry, f'/{robot}/odometry',
                                 self.on_ground_truth, qos_profile_sensor_data)

        if self.show_cam:
            cv2.namedWindow('down camera', cv2.WINDOW_NORMAL)
        if self.show_flow:
            cv2.namedWindow('optical flow', cv2.WINDOW_NORMAL)
        self.get_logger().info(
            f"flow_eval up, compare_frame='{self.frame}'. Publishing /eval/* "
            "(ground_truth, flow, ekf, pressure"
            + (", gtsam" if self.gtsam.available else "") + ")")

    # ---- sensor callbacks ----
    def on_imu(self, msg):
        # /imu/data is already ENU/FLU (imu_shim). For the GTSAM path we want body-frame
        # accel/gyro, which are frame-of-the-body regardless of world convention, so use
        # them directly. yaw is taken in the compare frame.
        q = msg.orientation
        self.gyro_body = (msg.angular_velocity.x, msg.angular_velocity.y,
                          msg.angular_velocity.z)
        self.accel_body = (msg.linear_acceleration.x, msg.linear_acceleration.y,
                           msg.linear_acceleration.z)
        # cache the body->NED quaternion (w,x,y,z) for gtsam gravity-aligned init.
        # /imu/data is ENU/FLU; convert its quat to NED/FRD via the tested frames path.
        self.last_quat_wxyz_ned = _enu_quat_to_ned_wxyz(q.x, q.y, q.z, q.w)
        yaw_enu = _yaw_from_quat_xyzw(q.x, q.y, q.z, q.w)
        # yaw in NED = 90 - yaw_enu wrapping; simplest correct route is to rotate a
        # heading vector. For planar integration only the world we integrate in matters:
        self.yaw_ned = np.pi / 2 - yaw_enu     # ENU yaw (CCW/East) -> NED yaw (CW/North)
        self.yaw = self.yaw_ned if self.frame == 'ned' else yaw_enu
        # feed GTSAM preintegration at IMU rate
        if self.gtsam.available:
            t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
            self.gtsam.add_imu(self.accel_body, self.gyro_body, t)

    def on_pressure(self, msg):
        # depth from gauge pressure (rho*g matches the scene; see depth_shim)
        self.depth = msg.fluid_pressure / (1000.0 * 9.81)
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        pz = depth_to_world_z(self.depth, self.frame)
        self.ekf.update_depth(pz, t)
        # /eval/pressure: depth only, x=y=0
        ax, ay, az = self._anchor(0.0, 0.0, pz)
        self._publish('pressure', ax, ay, az, msg.header.stamp)

    def on_altitude(self, msg):
        self.altitude = msg.data

    def on_ground_truth(self, msg):
        # Stonefish odometry: NED world position. Convert to compare frame.
        p = msg.pose.pose.position
        pos_ned = np.array([p.x, p.y, p.z])
        px, py, pz = gt_world_to_compare(pos_ned, self.frame)
        if self.gt_anchor is None:
            # All estimators dead-reckon from ZERO, but ground truth starts at the spawn
            # (start_position="-12.1 0 0.3" in sauvc_qualification.scn). Anchor every track
            # to ground truth's first pose so the comparison is displacement-from-start,
            # which is what dead reckoning actually measures. Without this the plots carry
            # a fixed ~12.1 m x-offset and can never overlay.
            self.gt_anchor = (px, py, pz)
        self._publish('ground_truth', px, py, pz, msg.header.stamp)

    def on_image(self, msg):
        if self.bridge is None:
            return
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        frame_bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        dt = 0.0 if self.prev_img_t is None else (t - self.prev_img_t)
        self.prev_img_t = t

        # gyro in camera frame: down camera x=image right, y=image down.
        # body FLU -> camera: for a down-looking camera, wx_cam ~ body pitch rate, etc.
        # Reuse the same convention flow_core expects: pass (wx, wy) body as camera rates.
        gyro_xy_cam = (self.gyro_body[0], self.gyro_body[1])
        alt = self.altitude if self.altitude is not None else max(
            self.get_parameter('pool_depth').value - self.depth, 0.1)

        res = self.flow.estimate(gray, dt, gyro_xy_cam, alt)
        if res is not None:
            # body velocity -> EKF + integrate to position
            # self.ekf.update_flow(res['vx'], res['vy'], self.yaw, t)
            # fx_, fy_ = self.flow_pos.update(res['vx'], res['vy'], self.yaw, t)
            # flow_core outputs body FLU. yaw_ned rotates an FRD vector; yaw_enu an FLU one.
            # Planar case: FLU->FRD is just vy -> -vy (eval_common.flu_to_frd_vec).
            vx_b = res['vx']
            vy_b = -res['vy'] if self.frame == 'ned' else res['vy']

            self.ekf.update_flow(vx_b, vy_b, self.yaw, t)
            fx_, fy_ = self.flow_pos.update(vx_b, vy_b, self.yaw, t)
            
            fz = depth_to_world_z(self.depth, self.frame)
            ax, ay, az = self._anchor(fx_, fy_, fz)
            self._publish('flow', ax, ay, az, msg.header.stamp,
                          vx=res['vx'], vy=res['vy'])

            # EKF publish (position + velocity)
            ex, ey, ez = self.ekf.position
            evx, evy = self.ekf.velocity
            aex, aey, aez = self._anchor(ex, ey, ez)
            self._publish('ekf', aex, aey, aez, msg.header.stamp, vx=evx, vy=evy)

            # GTSAM path: independent metric velocity, integrated
            if self.gtsam.available:
                # Rotate flow body velocity into NED world (the graph runs in NED).
                c, sn = np.cos(self.yaw_ned), np.sin(self.yaw_ned)
                vy_frd = -res['vy']                      # FLU -> FRD, graph world is NED
                fv_ned = np.array([c * res['vx'] - sn * vy_frd,
                                    sn * res['vx'] + c * vy_frd, 0.0])
                # fv_ned = np.array([c * res['vx'] - sn * res['vy'],
                #                    sn * res['vx'] + c * res['vy'], 0.0])
                if not self.gtsam.initialized:
                    # Seed attitude from the AHRS (gravity-aligned) + initial flow velocity.
                    q = self.last_quat_wxyz_ned
                    self.gtsam.initialize(q, fv_ned, self.depth)
                else:
                    out = self.gtsam.add_keyframe(fv_ned, self.depth)
                    if out is not None:
                        pos_ned, vel_ned = out
                        # graph pos is NED absolute (z=depth). Convert to compare frame,
                        # then anchor x/y to ground truth start like the others.
                        gp = gt_world_to_compare(np.array(pos_ned), self.frame)
                        gx, gy, gz = self._anchor(gp[0], gp[1], gp[2])
                        self._publish('gtsam', gx, gy, gz, msg.header.stamp,
                                      vx=float(vel_ned[0]), vy=float(vel_ned[1]))

        if self.show_cam:
            cv2.imshow('down camera', frame_bgr)
        if self.show_flow:
            cv2.imshow('optical flow', self.flow.overlay(frame_bgr))
        if self.show_cam or self.show_flow:
            cv2.waitKey(1)

    def _anchor(self, x, y, z):
        # subtract ground truth's start pose so every track begins where GT begins.
        if self.gt_anchor is None:
            return x, y, z
        return x + self.gt_anchor[0], y + self.gt_anchor[1], z + self.gt_anchor[2]

    def _publish(self, key, x, y, z, stamp, vx=0.0, vy=0.0):
        self._latest[key] = (x, y, z)
        self._maybe_print(stamp)
        od = Odometry()
        od.header.stamp = stamp
        od.header.frame_id = 'map_' + self.frame
        od.child_frame_id = 'base_link'
        od.pose.pose.position.x = float(x)
        od.pose.pose.position.y = float(y)
        od.pose.pose.position.z = float(z)
        od.twist.twist.linear.x = float(vx)
        od.twist.twist.linear.y = float(vy)
        self.pubs[key].publish(od)

    def _maybe_print(self, stamp):
        if not self.print_est:
            return
        t = stamp.sec + stamp.nanosec * 1e-9
        if t - self._last_print < self.print_period:
            return
        self._last_print = t

        def fmt(v):
            if v is None:
                return f"{'--':>8} {'--':>8} {'--':>8}"
            return f"{v[0]:+8.3f} {v[1]:+8.3f} {v[2]:+8.3f}"

        order = ['ground_truth', 'flow', 'ekf', 'pressure', 'gtsam']
        lines = [f"\n─ estimates [{self.frame.upper()} frame]  "
                 f"x        y        z ─────────────"]
        for k in order:
            if k == 'gtsam' and not self.gtsam.available:
                continue
            lines.append(f"  {k:<13} {fmt(self._latest[k])}")
        # quick sign sanity: does /eval/flow vx agree with ground-truth motion?
        gt, fl = self._latest['ground_truth'], self._latest['flow']
        print('\n'.join(lines))


def main():
    rclpy.init()
    node = FlowEvalNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node.show:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
