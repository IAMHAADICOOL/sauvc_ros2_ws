#!/usr/bin/env python3
"""rtf_monitor_node — measures the simulator's real-time factor and shouts if it isn't 1.

READ THIS BEFORE TRUSTING ANY VELOCITY NUMBER FROM THE SIM.

The problem
-----------
`stonefish_ros2/src/stonefish_ros2/ROS2Interface.cpp` stamps EVERY message with
`nh_->get_clock()->now()` -- the wall clock at publish time. The sample's own simulation
timestamp is thrown away: across the whole file there are 20 uses of `get_clock()->now()`
and 0 uses of `s.getTimestamp()`. (The ROS1 package does the opposite -- it uses
`ros::Time(s.getTimestamp())` -- so do not reason from the ROS1 source here.) There is
also no /clock publisher, which is why `ros2 topic hz /clock` reports nothing and why
`use_sim_time` must stay FALSE everywhere.

Now follow the consequence through `flow_velocity_node`:

    t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
    dt = t - self.last_stamp

`dt` is therefore WALL-CLOCK elapsed time. But the pixels in those two frames were
rendered from physics that advanced by SIMULATION time. Sensor `rate="30.0"` in the
scene means 30 samples per SIMULATED second. So if the simulator runs at real-time
factor R:

    physical displacement between frames = v_true * (1/30)      [sim seconds]
    wall-clock dt between frames         = (1/30) / R           [wall seconds]
    flow's reported velocity             = v_true * R

**Optical-flow velocity is scaled by exactly the real-time factor.**

Depth is not (it is a direct measurement). Gyro rates are not (physics quantities,
reported as-is). DVL velocity is not. So at R != 1 the sim is not merely slow -- it is
KINEMATICALLY INCONSISTENT: the EKF would fuse true angular rates against
R-scaled linear velocity, and flow_scorer would report a scale error of exactly R and
invite you to "fix" a calibration that was never broken.

With two 1280x720 cameras at 30 Hz plus Stonefish's underwater rendering, R < 1 is the
default expectation on most GPUs, not an edge case. R = 1 is a hard prerequisite for the
localization stack in sim, not a nice-to-have.

How this node measures R
------------------------
Two independent estimators, because each fails in a different place:

  1. RATE method (always available): the scene declares each sensor's rate in SIM time,
     so the observed wall-clock publish rate is `declared_rate * R`. Robust, works while
     stationary, but depends on you telling it the declared rate.

  2. KINEMATIC method (needs motion): differentiate ground-truth POSITION over wall-clock
     stamps and compare its magnitude to the ground-truth TWIST magnitude, which is
     reported directly in physics units. The ratio is R. Magnitudes are frame-invariant,
     so this needs no NED/ENU conversion and no assumption about whether the twist is
     body or world. Only meaningful above `min_speed`.

Both are read from /sauvc_auv/odometry, which is ground truth -- fine, because this node
is a DIAGNOSTIC and never feeds the estimator. It publishes /sim/rtf for logging and
warns on a throttle if R strays outside tolerance.

If R < 1, fix the simulator, do not fudge the numbers: drop camera resolution or rate in
my_auv.scn, lower `rendering_quality` in the launch file, or run the headless build.
"""

import math
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from nav_msgs.msg import Odometry


class RtfMonitorNode(Node):
    def __init__(self):
        super().__init__('rtf_monitor_node')
        self.declare_parameter('odom_topic', '/sauvc_auv/odometry')
        # MUST match rate="..." on the odometry sensor in my_auv.scn.
        self.declare_parameter('declared_odom_rate', 30.0)
        self.declare_parameter('window', 90)          # samples (~3 s at 30 Hz)
        self.declare_parameter('min_speed', 0.05)     # m/s, below this skip kinematic
        self.declare_parameter('tolerance', 0.05)     # warn if |R-1| > this
        self.declare_parameter('report_period', 5.0)  # s

        g = lambda n: self.get_parameter(n).value
        self.declared_rate = g('declared_odom_rate')
        self.min_speed = g('min_speed')
        self.tol = g('tolerance')

        n = int(g('window'))
        self.stamps = deque(maxlen=n)
        self.pos = deque(maxlen=n)
        self.ratios = deque(maxlen=n)

        self.pub_rtf = self.create_publisher(Float32, '/sim/rtf', 10)
        self.create_subscription(Odometry, g('odom_topic'), self.on_odom, 20)
        self.create_timer(g('report_period'), self.report)
        self.get_logger().info(
            f"rtf_monitor: watching {g('odom_topic')} (declared {self.declared_rate} Hz sim-time). "
            'Stonefish stamps with WALL CLOCK, so optical-flow velocity scales with RTF.')

    def on_odom(self, msg):
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = np.array([msg.pose.pose.position.x,
                      msg.pose.pose.position.y,
                      msg.pose.pose.position.z])
        v = math.sqrt(msg.twist.twist.linear.x ** 2 +
                      msg.twist.twist.linear.y ** 2 +
                      msg.twist.twist.linear.z ** 2)

        if self.stamps:
            dt = t - self.stamps[-1]
            if dt > 1e-6 and v > self.min_speed:
                # |d(pos)/dt_wall| / |twist_physics| == RTF. Magnitudes only, so the
                # body-vs-world frame of the twist is irrelevant here.
                apparent = np.linalg.norm(p - self.pos[-1]) / dt
                self.ratios.append(apparent / v)

        self.stamps.append(t)
        self.pos.append(p)

    def _rate_rtf(self):
        if len(self.stamps) < 5:
            return None
        span = self.stamps[-1] - self.stamps[0]
        if span <= 1e-6:
            return None
        observed = (len(self.stamps) - 1) / span
        return observed / self.declared_rate

    def _kinematic_rtf(self):
        if len(self.ratios) < 10:
            return None
        return float(np.median(self.ratios))   # median: robust to render hitches

    def report(self):
        r_rate = self._rate_rtf()
        r_kin = self._kinematic_rtf()
        if r_rate is None:
            self.get_logger().warn('rtf_monitor: no odometry yet — is the sim running?')
            return

        best = r_kin if r_kin is not None else r_rate
        self.pub_rtf.publish(Float32(data=float(best)))

        kin_txt = f'{r_kin:.3f}' if r_kin is not None else 'n/a (vehicle too slow)'
        msg = f'RTF: rate-method {r_rate:.3f} | kinematic-method {kin_txt}'

        if abs(best - 1.0) > self.tol:
            self.get_logger().warn(
                f'{msg}  <-- REAL-TIME FACTOR IS NOT 1. '
                f'Optical-flow velocity is being scaled by ~{best:.3f}x, while gyro rates '
                'and the DVL are NOT. Your EKF is fusing inconsistent kinematics and '
                'flow_scorer will report a bogus ~'
                f'{(best - 1.0) * 100:+.0f}% scale error. Reduce camera resolution/rate in '
                'my_auv.scn or rendering_quality in the launch file until this reads 1.00.')
        else:
            self.get_logger().info(msg)

        if r_kin is not None and abs(r_kin - r_rate) > 0.1:
            self.get_logger().warn(
                f'rtf_monitor: the two methods disagree ({r_rate:.3f} vs {r_kin:.3f}). '
                'Check that declared_odom_rate matches rate="..." on the odometry sensor '
                'in my_auv.scn.')


def main():
    rclpy.init()
    rclpy.spin(RtfMonitorNode())


if __name__ == '__main__':
    main()
