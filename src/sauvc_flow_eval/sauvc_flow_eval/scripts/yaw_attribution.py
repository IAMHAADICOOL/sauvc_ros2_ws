#!/usr/bin/env python3
"""yaw_attribution.py — is the flow y-drift the IMU/shim or the camera mount?

Standalone, no build. Run while the sim is up and drive a genuine straight leg:
  python3 yaw_attribution.py --ros-args -p robot_name:=sauvc_auv

Prints, per GT message:
  imu_yaw_ned : from /imu/data (ENU/FLU, shimmed)  ->  pi/2 - yaw_enu
  gt_yaw_ned  : from /<robot>/odometry orientation (FRD->NED)  ->  its yaw directly
  gt_vel_hdg  : atan2(dy,dx) of GT position (only shown while translating)
  instant     : per-message diff (NOISY — see below)
  settled     : SPEED-GATED CIRCULAR MEAN of imu-gt, accumulated only while the
                vehicle is actually translating above min_speed. This is the number
                that means something; the instantaneous column does not.

WHY 'settled' AND NOT THE RAW DIFF: at rest, or during spin/turn/collision transients,
imu_yaw_ned and gt_yaw_ned can differ by SEVERAL DEGREES from real vehicle dynamics
(bobbing, thruster ramp-up, PID hunting) that have nothing to do with a fixed
sensor/mount bias. A one-off instantaneous reading cannot separate that noise from a
small constant offset. Only trust 'settled' once its sample count is a few dozen AND
gt_vel_hdg has been roughly constant (a real straight leg) over that time — if the
heading is swinging around during accumulation, drive a cleaner straight leg and
restart this script.

Read the settled number once it has enough samples:
  * settled ~= 0                    -> IMU yaw is CORRECT. The offset lives in the
      DOWN-CAMERA mount yaw (flow body frame). Check the camera sensor rpy in the
      vehicle .scn (should be exactly 0 0 1.5708).
  * settled ~= the flow-autocal offset (e.g. ~+2.6 deg) -> IMU yaw is WRONG. It's the
      IMU path: either the imu_shim ENU<->NED conversion, or the IMU sensor's own
      mount yaw in the .scn.

STALENESS CHECK: if consecutive odometry messages carry the identical header stamp,
that means the topic isn't actually publishing new data (paused sim, stuck bridge) —
this script will WARN once rather than silently re-printing frozen numbers forever.
"""
import math
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from nav_msgs.msg import Odometry


def yaw_of(qx, qy, qz, qw):
    return math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class YawAttribution(Node):
    def __init__(self):
        super().__init__('yaw_attribution')
        self.declare_parameter('robot_name', 'sauvc_auv')
        self.declare_parameter('print_rate', 2.0)
        self.declare_parameter('min_speed', 0.05)   # m/s, only accumulate above this
        robot = self.get_parameter('robot_name').value
        self.min_speed = float(self.get_parameter('min_speed').value)
        self.imu_yaw = None
        self.gt_yaw = None
        self.gt_hdg = None
        self._prev = None
        self._last = 0.0
        self.period = 1.0 / max(self.get_parameter('print_rate').value, 0.1)
        # speed-gated circular mean of (imu_yaw - gt_yaw), same method flow_eval_node
        # uses internally for flow_yaw_offset — this is the number that isn't noise.
        self._s = 0.0
        self._c = 0.0
        self._n = 0
        # staleness guard
        self._last_stamp = None
        self._stale_warned = False
        self.create_subscription(Imu, '/imu/data', self.on_imu, qos_profile_sensor_data)
        self.create_subscription(Odometry, f'/{robot}/odometry', self.on_odom,
                                 qos_profile_sensor_data)
        self.get_logger().info(f'listening: /imu/data + /{robot}/odometry '
                               f'(min_speed={self.min_speed} m/s for the settled estimate)')

    def on_imu(self, m):
        q = m.orientation
        self.imu_yaw = math.pi / 2 - yaw_of(q.x, q.y, q.z, q.w)   # ENU -> NED, as the node does

    def on_odom(self, m):
        stamp = (m.header.stamp.sec, m.header.stamp.nanosec)
        if self._last_stamp is not None and stamp == self._last_stamp and not self._stale_warned:
            self._stale_warned = True
            self.get_logger().warn(
                'odometry header stamp is NOT advancing (identical stamp received twice). '
                'The topic appears frozen — check whether the sim is paused, the bridge '
                'has stalled, or the process has hung. Numbers below are stale until this '
                'clears; they are NOT evidence of anything about the yaw offset.')
        self._last_stamp = stamp

        q = m.pose.pose.orientation
        self.gt_yaw = yaw_of(q.x, q.y, q.z, q.w)                  # FRD->NED quat: yaw is NED heading
        p = m.pose.pose.position
        t = m.header.stamp.sec + m.header.stamp.nanosec * 1e-9
        speed = 0.0
        if self._prev is not None:
            dt = t - self._prev[2]
            dx, dy = p.x - self._prev[0], p.y - self._prev[1]
            if 1e-3 < dt < 1.0:
                speed = math.hypot(dx, dy) / dt
                if speed > self.min_speed:
                    self.gt_hdg = math.atan2(dy, dx)
        self._prev = (p.x, p.y, t)

        # accumulate the settled estimate ONLY while genuinely translating — this is
        # what filters out the at-rest / spin-transient noise you saw in the raw diff.
        if speed > self.min_speed and self.imu_yaw is not None and self.gt_yaw is not None:
            d = wrap(self.imu_yaw - self.gt_yaw)
            self._s += math.sin(d)
            self._c += math.cos(d)
            self._n += 1

        self._report(t)

    def _report(self, t):
        if t - self._last < self.period or self.imu_yaw is None or self.gt_yaw is None:
            return
        self._last = t
        d = lambda a, b: (math.degrees(wrap(a - b))
                          if (a is not None and b is not None) else float('nan'))
        hdg = 'n/a (still)' if self.gt_hdg is None else f'{math.degrees(self.gt_hdg):+7.2f}'
        settled = (f'{math.degrees(math.atan2(self._s, self._c)):+6.2f} deg (n={self._n})'
                  if self._n > 0 else '-- (n=0, no straight-leg samples yet)')
        print(f"imu_yaw_ned={math.degrees(self.imu_yaw):+7.2f}  "
              f"gt_yaw_ned={math.degrees(self.gt_yaw):+7.2f}  gt_vel_hdg={hdg}  "
              f"|  instant={d(self.imu_yaw, self.gt_yaw):+6.2f} deg (noisy)  "
              f"settled={settled}")


def main():
    rclpy.init()
    try:
        rclpy.spin(YawAttribution())
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
