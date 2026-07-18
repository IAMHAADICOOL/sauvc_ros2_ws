#!/usr/bin/env python3
"""direct_control_node — PATH A: /cmd/setpoint -> PID -> mixer -> Stonefish thrusters.

This is the NO-ArduSub control path. It closes velocity and depth loops directly in ROS2
and writes thruster setpoints straight to the simulator. Use it to develop and test the
localization + mission stack without bringing SITL into the picture — fewer moving parts,
deterministic, and every gain is visible in one YAML.

    /cmd/setpoint (Twist, body)         from mission_node — desired vx, vy, vz, yaw-rate
    /odometry/filtered (Odometry)       from the EKF — measured body velocities
    /depth (PoseWithCovarianceStamped)  from depth_shim — measured depth (z = -depth)
      -> PID per axis -> ThrusterMixer -> /sauvc_auv/thruster_setpoints (Float64MultiArray)

WHICH AXES ARE VELOCITY vs POSITION
  surge (vx), sway (vy), yaw-rate : velocity loops. mission_node commands body velocities
                                    directly, so the PID tracks commanded vs EKF-measured
                                    velocity.
  heave (vz)                      : mission_node's Twist.linear.z is treated as a DEPTH
                                    RATE command by default (cmd_z_is_depth:=false), OR as
                                    an absolute depth setpoint (cmd_z_is_depth:=true) if you
                                    would rather command "go to 1.0 m". The finals sequence
                                    (DIVE to cruise_depth, then hold) wants absolute depth,
                                    so mission integration will likely flip this true — but
                                    the default matches the Twist semantics literally.

WHY THIS LIVES IN sauvc_sim_bridge, NOT sauvc_mission
  On the real robot this whole node is replaced by ArduSub: mission_node's /cmd/setpoint
  goes to the autopilot, which runs its own control loops and mixer. So a direct PID
  controller is SIM INFRASTRUCTURE, not part of the portable stack — exactly the kind of
  thing the shim package exists to hold. Path B (ardusub_setpoint_node) is the
  higher-fidelity twin that DOES go through the autopilot; this is the fast path.

  Crucially, mission_node is byte-identical either way: it emits /cmd/setpoint and never
  knows which controller consumed it. That is the seam.

FRAMES: /cmd/setpoint and /odometry/filtered twist are body-frame ENU/FLU (your stack's
convention). The mixer wants body FRD. vx (forward) is the same in both; vy and yaw flip
sign FLU->FRD. That conversion happens HERE, explicitly, right before the mixer — the one
place body ENU meets body FRD.
"""

import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped
from std_msgs.msg import Float64MultiArray

from sauvc_sim_bridge.control_core import ThrusterMixer, PID


class DirectControlNode(Node):
    def __init__(self):
        super().__init__('direct_control_node')
        self.declare_parameter('robot_name', 'sauvc_auv')
        self.declare_parameter('rate_hz', 30.0)
        self.declare_parameter('cmd_timeout', 0.5)   # s; zero thrust if no command
        self.declare_parameter('cmd_z_is_depth', False)

        # Gains — PLACEHOLDERS. Tune in sim (surge/yaw first, then depth). The vehicle is
        # small and the thrusters are strong, so start gentle.
        self.declare_parameter('surge_kp', 1.5); self.declare_parameter('surge_ki', 0.3); self.declare_parameter('surge_kd', 0.0)
        self.declare_parameter('sway_kp', 1.5);  self.declare_parameter('sway_ki', 0.3);  self.declare_parameter('sway_kd', 0.0)
        self.declare_parameter('yaw_kp', 1.0);   self.declare_parameter('yaw_ki', 0.1);   self.declare_parameter('yaw_kd', 0.0)
        self.declare_parameter('depth_kp', 3.0); self.declare_parameter('depth_ki', 0.5); self.declare_parameter('depth_kd', 0.8)

        g = lambda n: self.get_parameter(n).value
        self.cmd_z_is_depth = g('cmd_z_is_depth')
        self.cmd_timeout = g('cmd_timeout')

        self.mixer = ThrusterMixer()
        self.get_logger().info(f'mixer condition number = {self.mixer.cond:.2f} '
                               '(lower is better; ~3 is healthy)')

        self.surge = PID(g('surge_kp'), g('surge_ki'), g('surge_kd'))
        self.sway  = PID(g('sway_kp'),  g('sway_ki'),  g('sway_kd'))
        self.yaw   = PID(g('yaw_kp'),   g('yaw_ki'),   g('yaw_kd'))
        self.depth = PID(g('depth_kp'), g('depth_ki'), g('depth_kd'))

        self.cmd = None            # latest Twist
        self.cmd_stamp = None
        self.meas_v = (0.0, 0.0)   # body vx, vy (FLU) from EKF
        self.meas_yawrate = 0.0
        self.meas_depth = 0.0

        robot = g('robot_name')
        self.pub = self.create_publisher(
            Float64MultiArray, f'/{robot}/thruster_setpoints', 10)
        self.create_subscription(Twist, '/cmd/setpoint', self.on_cmd, 10)
        self.create_subscription(Odometry, '/odometry/filtered', self.on_odom, 10)
        self.create_subscription(PoseWithCovarianceStamped, '/depth', self.on_depth, 10)

        self.dt = 1.0 / g('rate_hz')
        self.create_timer(self.dt, self.tick)
        self.get_logger().info(
            f'direct_control up -> /{robot}/thruster_setpoints @ {g("rate_hz")} Hz. '
            f'cmd_z_is_depth={self.cmd_z_is_depth}')

    def on_cmd(self, msg):
        self.cmd = msg
        self.cmd_stamp = self.get_clock().now()

    def on_odom(self, msg):
        # EKF twist is body-frame FLU. Store as-is; convert at the mixer.
        self.meas_v = (msg.twist.twist.linear.x, msg.twist.twist.linear.y)
        self.meas_yawrate = msg.twist.twist.angular.z

    def on_depth(self, msg):
        self.meas_depth = -msg.pose.pose.position.z   # z = -depth -> depth

    def tick(self):
        # Fail safe: no fresh command -> coast to neutral.
        if self.cmd is None or self.cmd_stamp is None:
            self.pub.publish(Float64MultiArray(data=[0.0] * 8))
            return
        age = (self.get_clock().now() - self.cmd_stamp).nanoseconds * 1e-9
        if age > self.cmd_timeout:
            self.pub.publish(Float64MultiArray(data=[0.0] * 8))
            return

        # --- velocity loops (body FLU) ---
        fx = self.surge.update(self.cmd.linear.x, self.meas_v[0], self.dt)
        fy_flu = self.sway.update(self.cmd.linear.y, self.meas_v[1], self.dt)
        mz_flu = self.yaw.update(self.cmd.angular.z, self.meas_yawrate, self.dt)

        # --- heave / depth loop ---
        if self.cmd_z_is_depth:
            # Twist.linear.z carries an ABSOLUTE depth setpoint [m, positive down].
            fz = self.depth.update(self.cmd.linear.z, self.meas_depth, self.dt)
        else:
            # Twist.linear.z is a depth-RATE command; a simple proportional map to heave
            # force (no measured depth-rate signal to close a loop on cleanly).
            fz = max(-1.0, min(1.0, self.cmd.linear.z))

        # --- body FLU -> body FRD, the one place these frames meet ---
        # forward (x) unchanged; left(+y FLU) -> right(+y FRD) flips; yaw (up->down) flips.
        fy = -fy_flu
        mz = -mz_flu
        # fz: mixer Fz is +down (FRD). depth PID output is already +down (descend),
        # so no flip. See control_core thruster-order note.

        u = self.mixer.wrench_to_thrust(fx, fy, fz, mz)
        self.pub.publish(Float64MultiArray(data=[float(v) for v in u]))


def main():
    rclpy.init()
    rclpy.spin(DirectControlNode())


if __name__ == '__main__':
    main()
