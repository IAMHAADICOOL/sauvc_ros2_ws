#!/usr/bin/env python3
"""Depth-PID mission (direct thruster control, no ArduSub):

  1. DESCEND to `target_depth` using a PID on pressure-derived depth,
     commanding the 4 vertical thrusters
  2. HOLD depth for `hold_time`
  3. FORWARD -> RIGHT (sway) -> LEFT -> BACKWARD, each for `leg_time`,
     while the PID keeps fixing depth
  4. SURFACE and stop

Thruster order on /<robot>/thruster_setpoints (Float64MultiArray, [-1,1]):
  [0] HFP  [1] HFS  [2] HAP  [3] HAS   (horizontal, vectored 45 deg)
  [4] VFP  [5] VFS  [6] VAP  [7] VAS   (vertical; POSITIVE = thrust DOWN = descend)

Run (with the sim up):
  ros2 run sauvc_motion_demo depth_pid_mission
  ros2 run sauvc_motion_demo depth_pid_mission --ros-args -p target_depth:=1.2 -p kp:=2.0

Tuning: kp/ki/kd are conservative defaults for the ~24 kg vehicle with
thrust_coeff=0.01. If depth oscillates -> lower kp or raise kd; if it sags
below/above target persistently -> raise ki slightly.
"""
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import FluidPressure
from std_msgs.msg import Float64MultiArray

RHO_G = 1000.0 * 9.81


class DepthPidMission(Node):
    def __init__(self):
        super().__init__('depth_pid_mission')
        p = self.declare_parameter
        p('robot', 'sauvc_auv')
        p('target_depth', 1.0)      # [m]
        p('hold_time', 5.0)         # [s]
        p('leg_time', 5.0)          # [s]
        p('surge_cmd', 0.3)         # [-1,1] forward/back magnitude
        p('sway_cmd', 0.3)          # [-1,1] right/left magnitude
        p('kp', 1.5)
        p('ki', 0.05)
        p('kd', 0.8)
        robot = self.get_parameter('robot').value

        self.pub = self.create_publisher(
            Float64MultiArray, f'/{robot}/thruster_setpoints', 10)
        self.create_subscription(FluidPressure, f'/{robot}/pressure',
                                 self.pressure_cb, qos_profile_sensor_data)

        self.p_ref = None
        self.depth = 0.0
        self.prev_depth = 0.0
        self.integral = 0.0
        self.state = 'WAIT'
        self.state_t0 = self.now()
        self.settled_since = None
        self.dt = 0.05
        self.timer = self.create_timer(self.dt, self.step)
        self.get_logger().info('mission started: waiting for pressure...')

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    def pressure_cb(self, msg):
        if self.p_ref is None:
            self.p_ref = msg.fluid_pressure
            self.get_logger().info(f'surface pressure reference: {self.p_ref:.1f} Pa')
        self.depth = (msg.fluid_pressure - self.p_ref) / RHO_G

    def depth_pid(self, target):
        e = target - self.depth                       # >0 -> need to go DOWN
        self.integral = max(-2.0, min(2.0, self.integral + e * self.dt))
        d = -(self.depth - self.prev_depth) / self.dt  # derivative on measurement
        self.prev_depth = self.depth
        kp = self.get_parameter('kp').value
        ki = self.get_parameter('ki').value
        kd = self.get_parameter('kd').value
        u = kp * e + ki * self.integral + kd * d
        return max(-0.6, min(0.6, u))                 # positive = descend

    def enter(self, state):
        self.get_logger().info(f'-> {state} (depth {self.depth:.2f} m)')
        self.state = state
        self.state_t0 = self.now()

    def step(self):
        if self.p_ref is None:
            return
        t = self.now() - self.state_t0
        target = self.get_parameter('target_depth').value
        hold = self.get_parameter('hold_time').value
        leg = self.get_parameter('leg_time').value
        surge = self.get_parameter('surge_cmd').value
        sway = self.get_parameter('sway_cmd').value

        h = [0.0, 0.0, 0.0, 0.0]   # HFP HFS HAP HAS
        v = self.depth_pid(target)

        if self.state == 'WAIT':
            self.enter('DESCEND')
        elif self.state == 'DESCEND':
            if abs(target - self.depth) < 0.10:
                if self.settled_since is None:
                    self.settled_since = self.now()
                elif self.now() - self.settled_since > 2.0:
                    self.settled_since = None
                    self.enter('HOLD')
            else:
                self.settled_since = None
        elif self.state == 'HOLD':
            if t > hold:
                self.enter('FORWARD')
        elif self.state == 'FORWARD':
            h = [surge, surge, surge, surge]
            if t > leg:
                self.enter('RIGHT')
        elif self.state == 'RIGHT':                    # sway starboard
            h = [sway, -sway, -sway, sway]
            if t > leg:
                self.enter('LEFT')
        elif self.state == 'LEFT':                     # sway port
            h = [-sway, sway, sway, -sway]
            if t > leg:
                self.enter('BACKWARD')
        elif self.state == 'BACKWARD':
            h = [-surge, -surge, -surge, -surge]
            if t > leg:
                self.enter('SURFACE')
        elif self.state == 'SURFACE':
            v = self.depth_pid(0.05)
            if self.depth < 0.15:
                self.enter('DONE')
        elif self.state == 'DONE':
            v = 0.0
            if t > 1.0:
                self.pub.publish(Float64MultiArray(data=[0.0] * 8))
                self.get_logger().info('mission complete')
                raise SystemExit

        self.pub.publish(Float64MultiArray(data=h + [v, v, v, v]))
        print(f'\r[{self.state:8s}] depth {self.depth:+.2f} m  v_cmd {v:+.2f}',
              end='', flush=True)


def main():
    rclpy.init()
    try:
        rclpy.spin(DepthPidMission())
    except (SystemExit, KeyboardInterrupt):
        pass


if __name__ == '__main__':
    main()
