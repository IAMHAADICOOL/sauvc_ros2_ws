#!/usr/bin/env python3
"""keyboard_teleop_node — 4-DOF keyboard teleop with depth-hold, publishing /cmd/setpoint.

Publishes the SAME topic mission_node uses (geometry_msgs/Twist on /cmd/setpoint, body
FLU) -- it is a stand-in for the mission FSM, and drives whichever controller is running
underneath (direct_control_node, Path A, or ardusub_setpoint_node, Path B) without
knowing which. All the actual key -> command logic lives in teleop_core.py, pure and
unit-tested; this file is ROS plumbing + raw-terminal keyboard reading around it.

RUN THIS IN ITS OWN TERMINAL, NOT INSIDE A COMBINED `ros2 launch`
------------------------------------------------------------------
This node puts the TTY into raw mode to read single keystrokes without Enter. That only
works cleanly when it owns the terminal outright. Under `ros2 launch` with other nodes
sharing the same console, stdin is usually not forwarded the way an interactive teleop
needs, and output from other nodes will scribble over the status line. Bring up the rest
of the stack with the launch files below, then run this SEPARATELY:

    Terminal 1: ros2 launch sauvc_teleop teleop_direct.launch.py     # Path A
    Terminal 2: ros2 run sauvc_teleop keyboard_teleop_node --ros-args -p depth_mode:=absolute

    Terminal 1: ros2 launch sauvc_teleop teleop_ardusub.launch.py    # Path B (+ your
                                                                       # SITL/json-bridge
                                                                       # processes)
    Terminal 2: ros2 run sauvc_teleop keyboard_teleop_node --ros-args -p depth_mode:=pulse

ONLY FOUR DOF ARE ACTUATED -- see teleop_core.py's docstring. There is no roll/pitch key
because control_core.ThrusterMixer does not control those axes.
"""

import sys
import time
import select

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped

from sauvc_teleop.teleop_core import (
    TeleopState, TeleopLimits, apply_key, seed_depth, command_twist)

try:
    import termios
    import tty
    _HAVE_TTY = True
except ImportError:      # pragma: no cover — non-POSIX platform
    _HAVE_TTY = False


HELP_TEMPLATE = """
sauvc_teleop -- 4 DOF only (surge/sway/yaw/depth). Roll & pitch are NOT actuated.
depth_mode = {mode}

  MOVEMENT (persists until changed)        DEPTH
    w/s : surge +/-                          r : shallower (up)
    a/d : sway  left/right                   f : deeper    (down)
    q/e : yaw   left/right (CCW/CW)          0 : surface [absolute mode only]

  space : zero surge/sway/yaw (depth hold is unaffected)
  x     : FULL STOP -- zero surge/sway/yaw; depth hold continues unchanged
  +/-   : bigger/smaller surge & sway step    [ / ] : bigger/smaller depth step
  CTRL-C: quit (publishes neutral on the way out)
"""


def _get_key(settings, timeout):
    """Non-blocking single-keystroke read. Returns '' on timeout (no key pressed)."""
    tty.setraw(sys.stdin.fileno())
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    key = sys.stdin.read(1) if rlist else ''
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


class KeyboardTeleopNode(Node):
    def __init__(self):
        super().__init__('keyboard_teleop_node')
        p = self.declare_parameter
        p('rate_hz', 20.0)
        p('surge_step', 0.15); p('sway_step', 0.15); p('yaw_step', 0.3)
        p('max_surge', 0.6);   p('max_sway', 0.6);    p('max_yaw', 1.2)
        p('depth_step', 0.1)
        p('min_depth', 0.0);   p('max_depth', 1.5)
        p('depth_mode', 'absolute')     # 'absolute' (Path A) | 'pulse' (Path B)
        p('pulse_duration', 0.6)        # s, pulse mode only
        p('pulse_rate', 0.3)            # m/s-equivalent sent during a pulse

        g = lambda n: self.get_parameter(n).value
        self.rate_hz = g('rate_hz')
        self.mode = g('depth_mode')
        if self.mode not in ('absolute', 'pulse'):
            raise ValueError(f"depth_mode must be 'absolute' or 'pulse', got {self.mode!r}")
        self.pulse_duration = g('pulse_duration')
        self.pulse_rate = g('pulse_rate')

        self.limits = TeleopLimits(max_surge=g('max_surge'), max_sway=g('max_sway'),
                                   max_yaw=g('max_yaw'), min_depth=g('min_depth'),
                                   max_depth=g('max_depth'))
        self.state = TeleopState(surge_step=g('surge_step'), sway_step=g('sway_step'),
                                 yaw_step=g('yaw_step'), depth_step=g('depth_step'))

        self.pub = self.create_publisher(Twist, '/cmd/setpoint', 10)
        self.create_subscription(PoseWithCovarianceStamped, '/depth', self.on_depth, 10)

        self.get_logger().info(
            f"keyboard_teleop up, depth_mode='{self.mode}'. Waiting for /depth to seed "
            "the initial hold target before publishing (avoids a lurch)..." )

        print(HELP_TEMPLATE.format(mode=self.mode))

    def on_depth(self, msg):
        d = -msg.pose.pose.position.z
        was_unset = self.state.depth_target is None
        seed_depth(self.state, d)
        if was_unset and self.state.depth_target is not None:
            self.get_logger().info(f'seeded depth target = {d:.2f} m from /depth')

    def status_line(self):
        dt = self.state.depth_target
        dt_s = f'{dt:5.2f}' if dt is not None else ' n/a '
        return (f'\rsurge {self.state.surge:+.2f}  sway {self.state.sway:+.2f}  '
               f'yaw {self.state.yaw:+.2f}  depth_target {dt_s}  '
               f'steps(v={self.state.surge_step:.2f} d={self.state.depth_step:.2f})  ')

    def publish(self):
        # In 'absolute' mode, don't publish anything until seeded — publishing z=0.0
        # before we know the real depth would command a lurch toward the surface.
        if self.mode == 'absolute' and self.state.depth_target is None:
            return
        vx, vy, vw, z = command_twist(self.state, self.mode, time.time(), self.pulse_rate)
        m = Twist()
        m.linear.x, m.linear.y, m.linear.z, m.angular.z = vx, vy, z, vw
        self.pub.publish(m)
        print(self.status_line(), end='', flush=True)


def main():
    if not _HAVE_TTY:
        raise RuntimeError('keyboard_teleop_node needs a POSIX TTY (termios/tty).')

    rclpy.init()
    node = KeyboardTeleopNode()
    settings = termios.tcgetattr(sys.stdin)
    dt = 1.0 / node.rate_hz
    try:
        while rclpy.ok():
            key = _get_key(settings, timeout=dt)
            if key == '\x03':          # Ctrl-C arrives as a raw byte in raw mode
                break
            if key:
                apply_key(node.state, key, node.limits, node.mode,
                         time.time(), node.pulse_duration)
            rclpy.spin_once(node, timeout_sec=0.0)   # flush the /depth subscription
            node.publish()
    except KeyboardInterrupt:
        pass
    finally:
        # Always leave the vehicle in neutral and the terminal usable.
        node.state.surge = node.state.sway = node.state.yaw = 0.0
        if node.mode == 'pulse':
            node.state.pulse_until = 0.0
        try:
            node.publish()
        except Exception:
            pass
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
        print('\nkeyboard_teleop: neutral published, exiting.')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
