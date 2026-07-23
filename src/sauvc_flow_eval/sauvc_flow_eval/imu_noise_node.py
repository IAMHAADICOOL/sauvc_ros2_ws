#!/usr/bin/env python3
"""imu_noise_node — physically CONSISTENT IMU corruption with known covariances.

WHY THIS EXISTS (replaces the .scn's <noise ... yaw_drift=...>):
  Stonefish's yaw_drift is a ramp added ONLY to the published orientation's yaw;
  the angular_velocity channel stays clean (docs list it as a separate "yaw angle
  drift rate"; confirmed against IMU.cpp and visible in the run logs: imu yaw error
  wanders while the gyro-integrated GTSAM yaw does not). That data is physically
  INCONSISTENT — gyros say "not rotating" while yaw creeps — so no estimator can
  ever model it: it is not in the gyro signal GTSAM preintegrates, and it is not a
  bias in any measured quantity. It is un-estimable BY CONSTRUCTION.

  A real magnetometer-less AHRS (HFI-A9 in a pool) drifts in yaw because yaw is the
  INTEGRAL OF THE GYRO: drift = integral(gyro bias + noise). Orientation and gyro
  agree with each other, and the bias IS estimable — that is precisely what the
  CombinedImuFactor's bias states are for. This node reproduces that mechanism:

      gyro_out  = gyro_true  + b_g(t) + N(0, gyro_noise_std)
      accel_out = accel_true + b_a    + N(0, accel_noise_std)
      b_g(t)    = b_g(0) + random walk        b_g(0) ~ N(0, gyro_bias_init_std)
      roll/pitch_out = true + N(0, rp_noise_std)      (gravity-referenced, bounded)
      yaw_out   = yaw_true(t0) + integral(wz_out dt)  (+ small white jitter)
                  -> drifts at exactly b_gz + noise random walk, CONSISTENTLY.

  The message covariance fields are filled with the SAME numbers, so consumers can
  read their R matrices straight off the message instead of hardcoding.

SETUP (three steps, nothing else changes):
  1. In my_auv.scn, zero the sim-side IMU noise so corruption happens in exactly ONE
     controlled, seeded place:
         <noise angle="0.0 0.0 0.0" angular_velocity="0.0"
                yaw_drift="0.0" linear_acceleration="0.0"/>
  2. Run this node (after the imu shim):
         ros2 run sauvc_flow_eval imu_noise_node          # or python3 imu_noise_node.py
     It subscribes /imu/data (the shim's ENU/FLU output) and publishes /imu/data_noisy.
  3. Point consumers at the noisy topic WITHOUT touching their code:
         ros2 run sauvc_flow_eval flow_eval_node --ros-args -r /imu/data:=/imu/data_noisy
     (Remap on the consumer only — this node itself keeps reading the clean topic.)

PARAMETER <-> FILTER ALIGNMENT (single source of truth):
     gyro_noise_std      <-> GtsamEstimator(gyro_sigma=...)            [0.0017]
     gyro_bias_rw_std    <-> GtsamEstimator(gyro_bias_rw=...)          [0.0001]
     accel_noise_std     <-> GtsamEstimator(accel_sigma=...)           [0.02]
     accel_bias_rw_std   <-> GtsamEstimator(accel_bias_rw=...)         [0.001]
     rp_noise_std        <-> GtsamEstimator(att_rp_sigma=...)          [0.01]
     yaw error over a correction window
                         <-> EkfEstimator(sigma_yaw=...) — compute it with the
                             existing estimate_covariance.py --drift-rate-deg-per-min
                             (deg/min = gyro_bias_init_std * 180/pi * 60).
  gyro_bias_init_std default 0.00029 rad/s reproduces the old .scn yaw_drift scale
  (~1 deg/min) — same magnitude of realism, now estimable.

SEEDED: same seed = identical bias draw and noise stream = reproducible runs. Change
the seed to get a "different unit of the same sensor".
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu


def _yaw_from_quat_xyzw(x, y, z, w):
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def _rp_from_quat_xyzw(x, y, z, w):
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    return roll, pitch


def _quat_xyzw_from_rpy(roll, pitch, yaw):
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    return (sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy)


class ImuNoiseNode(Node):
    def __init__(self):
        super().__init__('imu_noise_node')
        p = self.declare_parameter
        p('input_topic', '/imu/data')
        p('output_topic', '/imu/data_noisy')
        p('seed', 7)                       # same seed = same sensor unit, reproducible
        # --- gyro: the numbers that drive REALISTIC, ESTIMABLE yaw drift ---
        p('gyro_noise_std', 0.0017)        # rad/s white       <-> gtsam gyro_sigma
        p('gyro_bias_init_std', 0.00029)   # rad/s constant bias draw (~1 deg/min yaw)
        p('gyro_bias_rw_std', 0.0001)      # rad/s/sqrt(s) walk <-> gtsam gyro_bias_rw
        # --- accel ---
        p('accel_noise_std', 0.02)         # m/s^2 white       <-> gtsam accel_sigma
        p('accel_bias_init_std', 0.02)     # m/s^2 constant bias draw
        p('accel_bias_rw_std', 0.001)      # m/s^2/sqrt(s)     <-> gtsam accel_bias_rw
        # --- orientation output (the AHRS emulation) ---
        p('rp_noise_std', 0.002)           # rad white on roll/pitch (gravity-fused)
        p('yaw_jitter_std', 0.005)         # rad white ON TOP of the integrated yaw
        g = lambda n: self.get_parameter(n).value

        rng = np.random.default_rng(int(g('seed')))
        self.sg = float(g('gyro_noise_std'))
        self.sa = float(g('accel_noise_std'))
        self.srp = float(g('rp_noise_std'))
        self.syj = float(g('yaw_jitter_std'))
        self.rw_g = float(g('gyro_bias_rw_std'))
        self.rw_a = float(g('accel_bias_rw_std'))
        self.b_g = rng.normal(0.0, float(g('gyro_bias_init_std')), 3)
        self.b_a = rng.normal(0.0, float(g('accel_bias_init_std')), 3)
        self.rng = rng
        self.yaw = None                    # integrated (corrupted) yaw state
        self.t_prev = None

        self.pub = self.create_publisher(Imu, g('output_topic'), 10)
        self.create_subscription(Imu, g('input_topic'), self.on_imu,
                                 qos_profile_sensor_data)
        self.get_logger().info(
            f"imu_noise up: {g('input_topic')} -> {g('output_topic')} | "
            f"gyro bias drawn (rad/s): "
            f"[{self.b_g[0]:+.5f} {self.b_g[1]:+.5f} {self.b_g[2]:+.5f}] "
            f"(z-bias => yaw drift {math.degrees(self.b_g[2])*60:+.2f} deg/min) | "
            "REMINDER: zero the <noise .../> line in my_auv.scn so this node is the "
            "only corruption source.")

    def on_imu(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dt = 0.0 if self.t_prev is None else max(0.0, min(t - self.t_prev, 0.2))
        self.t_prev = t

        # bias random walks (per-axis, variance rw^2 * dt)
        if dt > 0.0:
            self.b_g += self.rng.normal(0.0, self.rw_g * math.sqrt(dt), 3)
            self.b_a += self.rng.normal(0.0, self.rw_a * math.sqrt(dt), 3)

        out = Imu()
        out.header = msg.header

        # ---- gyro + accel: truth + bias + white noise ----
        w = np.array([msg.angular_velocity.x, msg.angular_velocity.y,
                      msg.angular_velocity.z])
        a = np.array([msg.linear_acceleration.x, msg.linear_acceleration.y,
                      msg.linear_acceleration.z])
        w_n = w + self.b_g + self.rng.normal(0.0, self.sg, 3)
        a_n = a + self.b_a + self.rng.normal(0.0, self.sa, 3)
        (out.angular_velocity.x, out.angular_velocity.y,
         out.angular_velocity.z) = map(float, w_n)
        (out.linear_acceleration.x, out.linear_acceleration.y,
         out.linear_acceleration.z) = map(float, a_n)

        # ---- orientation: gravity-referenced roll/pitch + GYRO-INTEGRATED yaw ----
        q = msg.orientation
        roll_t, pitch_t = _rp_from_quat_xyzw(q.x, q.y, q.z, q.w)
        yaw_t = _yaw_from_quat_xyzw(q.x, q.y, q.z, q.w)
        if self.yaw is None:
            self.yaw = yaw_t               # start aligned with truth, drift from here
        else:
            # small-tilt yaw-rate ~ body z rate (FLU). Integrating the CORRUPTED
            # rate is the whole point: drift(t) = b_gz*t + noise random walk —
            # exactly what a bias-estimating filter can observe and remove.
            self.yaw += float(w_n[2]) * dt
        roll_o = roll_t + float(self.rng.normal(0.0, self.srp))
        pitch_o = pitch_t + float(self.rng.normal(0.0, self.srp))
        yaw_o = self.yaw + float(self.rng.normal(0.0, self.syj))
        (out.orientation.x, out.orientation.y,
         out.orientation.z, out.orientation.w) = _quat_xyzw_from_rpy(
            roll_o, pitch_o, yaw_o)

        # ---- covariances: publish the SAME numbers the corruption used, so any
        # consumer can read R off the message instead of hardcoding. Yaw variance is
        # the jitter only — the drift part is unbounded by design and belongs in the
        # consumer's process model / bias states, not in a static field. ----
        rp2, yj2 = self.srp ** 2, self.syj ** 2
        out.orientation_covariance = [rp2, 0.0, 0.0,
                                      0.0, rp2, 0.0,
                                      0.0, 0.0, yj2]
        g2 = self.sg ** 2
        out.angular_velocity_covariance = [g2, 0.0, 0.0,
                                           0.0, g2, 0.0,
                                           0.0, 0.0, g2]
        a2 = self.sa ** 2
        out.linear_acceleration_covariance = [a2, 0.0, 0.0,
                                              0.0, a2, 0.0,
                                              0.0, 0.0, a2]
        self.pub.publish(out)


def main():
    rclpy.init()
    node = ImuNoiseNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
