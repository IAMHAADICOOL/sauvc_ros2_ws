#!/usr/bin/env python3
"""ardusub_setpoint_node — PATH B: /cmd/setpoint -> MAVLink -> ArduSub SITL.

This is the HIGH-FIDELITY control path. mission_node's body-velocity command goes to the
ArduSub autopilot, which runs its own stabilization and mixer and drives the thrusters via
the JSON physics backend. The existing ardusub_json_bridge.py already closes the other
half of the loop (ArduSub servo PWM -> /sauvc_auv/thruster_setpoints, and Stonefish state
-> SITL). This node supplies the MISSING piece: getting mission_node's intent INTO ArduSub.

    /cmd/setpoint (Twist, body FLU)   from mission_node
      -> MANUAL_CONTROL (x,y,z,r) over MAVLink -> ArduSub SITL (udp:127.0.0.1:14550)

WHY MANUAL_CONTROL AND NOT SET_POSITION_TARGET
  ArduSub does not fly waypoints the way ArduCopter does; its bread-and-butter pilot input
  is MANUAL_CONTROL (the joystick message), interpreted in the active flight mode. Running
  ArduSub in STABILIZE or DEPTH_HOLD and feeding MANUAL_CONTROL gives you exactly the pilot
  experience the real vehicle has — which is the point of testing through SITL rather than
  bypassing it. DEPTH_HOLD is the useful one: its z channel becomes a depth-rate command
  with the autopilot holding depth when z is centered, so mission_node's descend/hold logic
  maps naturally.

  MANUAL_CONTROL axes are int16 in [-1000, 1000]:
    x = forward(+)/back    y = right(+)/left    z = throttle/heave    r = yaw rate(+CW)
  In DEPTH_HOLD, z=500 is neutral-hold and the vehicle maintains depth; <500 ascends,
  >500 descends. In STABILIZE, z is direct vertical throttle (500 neutral). This node
  exposes `z_neutral` so you can match whichever mode you arm in.

FRAME: /cmd/setpoint is body FLU. MANUAL_CONTROL is body FRD-ish pilot convention
(x forward, y right, positive r = yaw right/CW). So y and r flip sign FLU->pilot, the same
flip direct_control_node does before its mixer. Surge is unchanged.

SCALING: Twist is in m/s and rad/s; MANUAL_CONTROL is dimensionless [-1000,1000]. The
`*_scale` params set what commanded speed corresponds to full stick. These are NOT a
controller — ArduSub does the actual control — they are just joystick sensitivity. Tune so
cruise_speed (0.4 m/s) lands around half stick.

PREREQUISITES (all already in the sim workspace / its README):
  1. ArduSub SITL running and listening on the JSON backend (port 9002).
  2. ardusub_json_bridge.py running (couples SITL <-> Stonefish thrusters + state).
  3. This node, talking MAVLink to SITL on 14550.
  4. Vehicle ARMED and in a useful mode (DEPTH_HOLD recommended). `auto_arm:=true` will
     arm and set mode on startup; otherwise do it from QGroundControl / mavproxy.

This node needs pymavlink:  pip install pymavlink
"""

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist

try:
    from pymavlink import mavutil
except ImportError as e:      # pragma: no cover
    raise ImportError('ardusub_setpoint_node needs pymavlink: pip install pymavlink') from e


def _clamp_i16(v):
    return int(max(-1000, min(1000, v)))


class ArduSubSetpointNode(Node):
    def __init__(self):
        super().__init__('ardusub_setpoint_node')
        self.declare_parameter('url', 'udp:127.0.0.1:14550')
        self.declare_parameter('rate_hz', 20.0)
        self.declare_parameter('cmd_timeout', 0.5)
        self.declare_parameter('auto_arm', False)
        self.declare_parameter('mode', 'ALT_HOLD')      # ArduSub's depth-hold mode name
        self.declare_parameter('z_neutral', 500)        # 500 = hold in ALT_HOLD/DEPTH_HOLD
        # joystick sensitivity: commanded m/s (or rad/s) that maps to full stick (1000)
        self.declare_parameter('surge_scale', 0.8)      # m/s at full x
        self.declare_parameter('sway_scale', 0.8)       # m/s at full y
        self.declare_parameter('yaw_scale', 1.0)        # rad/s at full r
        self.declare_parameter('heave_scale', 0.5)      # m/s at full z deflection

        g = lambda n: self.get_parameter(n).value
        self.cmd_timeout = g('cmd_timeout')
        self.z_neutral = int(g('z_neutral'))
        self.surge_scale = g('surge_scale')
        self.sway_scale = g('sway_scale')
        self.yaw_scale = g('yaw_scale')
        self.heave_scale = g('heave_scale')

        url = g('url')
        self.get_logger().info(f'connecting to ArduSub at {url} …')
        self.mav = mavutil.mavlink_connection(url, source_system=255)
        self.mav.wait_heartbeat()
        self.get_logger().info(f'heartbeat from system {self.mav.target_system}')

        if g('auto_arm'):
            self._set_mode(g('mode'))
            self._arm()

        self.cmd = None
        self.cmd_stamp = None
        self.create_subscription(Twist, '/cmd/setpoint', self.on_cmd, 10)
        self.create_timer(1.0 / g('rate_hz'), self.tick)
        self.get_logger().info('ardusub_setpoint up: /cmd/setpoint -> MANUAL_CONTROL')

    def _set_mode(self, mode_name):
        mode_id = self.mav.mode_mapping().get(mode_name)
        if mode_id is None:
            self.get_logger().warn(f'unknown mode {mode_name!r}; leaving mode unchanged. '
                                   f'available: {list(self.mav.mode_mapping().keys())}')
            return
        self.mav.mav.set_mode_send(
            self.mav.target_system,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode_id)
        self.get_logger().info(f'requested mode {mode_name}')

    def _arm(self):
        self.mav.mav.command_long_send(
            self.mav.target_system, self.mav.target_component,
            mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0,
            1, 0, 0, 0, 0, 0, 0)
        self.get_logger().info('arm requested')

    def on_cmd(self, msg):
        self.cmd = msg
        self.cmd_stamp = self.get_clock().now()

    def _neutral(self):
        # x=y=r=0, z=hold. In ALT_HOLD this holds position/depth.
        self.mav.mav.manual_control_send(
            self.mav.target_system, 0, 0, self.z_neutral, 0, 0)

    def tick(self):
        if self.cmd is None or self.cmd_stamp is None:
            self._neutral()
            return
        age = (self.get_clock().now() - self.cmd_stamp).nanoseconds * 1e-9
        if age > self.cmd_timeout:
            self._neutral()
            return

        # body FLU -> pilot convention (x fwd, y right, r CW): flip y and yaw.
        x = _clamp_i16(self.cmd.linear.x / self.surge_scale * 1000.0)
        y = _clamp_i16(-self.cmd.linear.y / self.sway_scale * 1000.0)
        r = _clamp_i16(-self.cmd.angular.z / self.yaw_scale * 1000.0)
        # heave: deflection about the hold neutral. +linear.z (descend, +down) -> z>neutral.
        z = _clamp_i16(self.z_neutral + self.cmd.linear.z / self.heave_scale * 1000.0)
        z = int(max(0, min(1000, z)))

        self.mav.mav.manual_control_send(self.mav.target_system, x, y, z, r, 0)


def main():
    rclpy.init()
    rclpy.spin(ArduSubSetpointNode())


if __name__ == '__main__':
    main()
