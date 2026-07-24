#!/usr/bin/env python3
"""keyboard_teleop_node — 4-DOF keyboard teleop with depth-hold + planar/heading hold,
publishing /cmd/setpoint.

Publishes the SAME topic mission_node uses (geometry_msgs/Twist on /cmd/setpoint, body
FLU) -- it is a stand-in for the mission FSM, and drives whichever controller is running
underneath (direct_control_node, Path A, or ardusub_setpoint_node, Path B) without
knowing which. All the actual key -> command logic AND the hold controller live in
teleop_core.py, pure and unit-tested; this file is ROS plumbing + raw-terminal keyboard
reading around it.

WHAT'S NEW
----------
  * Per-keystroke increments (surge_step / sway_step / yaw_step) are runtime-settable via
    `ros2 param set` now, not only at launch. yaw_step's default is lowered to 0.10.
  * PLANAR + HEADING HOLD (absolute mode / Path A only). When you zero an axis, an outer
    position/heading PID pins the vehicle instead of letting it coast off with the waves:
      - yaw ~0            -> hold current heading
      - surge & sway ~0   -> hold current planar position (station-keep)
      - 'x' (full stop)   -> now actively brakes to a full station-keep, not just coast
    ArduSub does its own hold, so hold is DISABLED in 'pulse' mode automatically.
  * FEEDBACK SOURCE is a runtime-settable parameter: 'ekf' | 'eskf' | 'gtsam' |
    'ground_truth' | 'none'. Each maps to a topic + world frame; NED sources are
    converted to ENU through the workspace's one sanctioned path (sauvc_sim_bridge.frames)
    before the hold ever sees them.

RUN THIS IN ITS OWN TERMINAL, NOT INSIDE A COMBINED `ros2 launch`
------------------------------------------------------------------
This node puts the TTY into raw mode to read single keystrokes without Enter. That only
works cleanly when it owns the terminal outright. Bring up the rest of the stack with the
launch files below, then run this SEPARATELY:

    Terminal 1: ros2 launch sauvc_teleop teleop_direct.launch.py     # Path A
    Terminal 2: ros2 run sauvc_teleop keyboard_teleop_node --ros-args \
                    -p depth_mode:=absolute -p feedback_source:=ekf

    Terminal 1: ros2 launch sauvc_teleop teleop_ardusub.launch.py    # Path B (+ your
                                                                       # SITL/json-bridge
                                                                       # processes)
    Terminal 2: ros2 run sauvc_teleop keyboard_teleop_node --ros-args -p depth_mode:=pulse

Switch feedback / retune live from a third terminal:
    ros2 param set /keyboard_teleop_node feedback_source ground_truth
    ros2 param set /keyboard_teleop_node yaw_step 0.05
    ros2 param set /keyboard_teleop_node hold_yaw_kp 1.6

FEEDBACK AVAILABILITY (Path A): 'ekf' (/odometry/filtered) and 'ground_truth'
(/sauvc_auv/odometry) are live with just teleop_direct.launch.py. 'gtsam' (/eval/gtsam)
needs flow_eval_node running. 'eskf' has NO producer in the workspace yet -- the param is
wired to /eval/eskf so it works the day you add one; until then hold simply won't engage
(the vehicle coasts, exactly as before). If feedback is stale/absent, hold stays OFF and
you fly open-loop -- it never blocks you.

ONLY FOUR DOF ARE ACTUATED -- see teleop_core.py's docstring. There is no roll/pitch key
because control_core.ThrusterMixer does not control those axes.
"""

import sys
import time
import math
import select

import rclpy
from rclpy.node import Node
from rcl_interfaces.msg import SetParametersResult
from geometry_msgs.msg import Twist
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry

from sauvc_teleop.teleop_core import (
    TeleopState, TeleopLimits, HoldGains, HoldController, Feedback,
    apply_key, seed_depth, command_twist)

# The one sanctioned NED<->ENU conversion, shared with the rest of the stack.
from sauvc_sim_bridge.frames import ned_to_enu_vec, ned_frd_quat_to_enu_flu

try:
    import termios
    import tty
    _HAVE_TTY = True
except ImportError:      # pragma: no cover — non-POSIX platform
    _HAVE_TTY = False


# feedback_source -> (topic param name, frame param name). All are nav_msgs/Odometry.
_FEEDBACK_SOURCES = ('ekf', 'eskf', 'gtsam', 'ground_truth', 'none')


HELP_TEMPLATE = """
sauvc_teleop -- 4 DOF only (surge/sway/yaw/depth). Roll & pitch are NOT actuated.
depth_mode = {mode}   |   hold = {hold}   |   feedback = {fb}

  MOVEMENT (persists until changed)        DEPTH
    w/s : surge +/-                          r : shallower (up)
    a/d : sway  left/right                   f : deeper    (down)
    q/e : yaw   left/right (CCW/CW)          0 : surface [absolute mode only]

  space : zero surge/sway/yaw  -> HOLD engages (heading + station-keep)
  x     : FULL STOP -> brake to station-keep; depth hold continues unchanged
  +/-   : bigger/smaller surge & sway step    [ / ] : bigger/smaller depth step
  CTRL-C: quit (publishes neutral on the way out)

Hold (absolute mode only) pins heading when yaw~0 and planar position when surge&sway~0,
using the '{fb}' estimate. It needs fresh feedback; if none arrives, you fly open-loop.
"""


def _yaw_from_quat_xyzw(x, y, z, w):
    """Heading (CCW from +x) from an (x,y,z,w) quaternion. Frame-agnostic in the sense
    that whatever world the quaternion is expressed in, this is that world's yaw."""
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


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
        p('surge_step', 0.15); p('sway_step', 0.15); p('yaw_step', 0.10)
        p('max_surge', 0.6);   p('max_sway', 0.6);    p('max_yaw', 0.6)
        p('depth_step', 0.1)
        p('min_depth', 0.0);   p('max_depth', 1.5)
        p('depth_mode', 'absolute')     # 'absolute' (Path A) | 'pulse' (Path B)
        p('pulse_duration', 0.6)        # s, pulse mode only
        p('pulse_rate', 0.3)            # m/s-equivalent sent during a pulse

        # --- hold (request #2/#3) ---
        p('hold_enabled', True)         # absolute mode only; ignored in pulse mode
        p('hold_yaw_kp', 1.2); p('hold_yaw_ki', 0.0); p('hold_yaw_kd', 0.0)
        p('hold_pos_kp', 0.8); p('hold_pos_ki', 0.0); p('hold_pos_kd', 0.0)
        p('hold_engage_deadband', 1e-3)

        # --- feedback (request #4) ---
        p('feedback_source', 'ekf')     # ekf | eskf | gtsam | ground_truth | none
        p('feedback_timeout', 0.5)      # s; older than this -> hold won't engage
        p('ekf_topic', '/odometry/filtered');        p('ekf_frame', 'enu')
        p('eskf_topic', '/eval/eskf');               p('eskf_frame', 'ned')
        p('gtsam_topic', '/eval/gtsam');             p('gtsam_frame', 'ned')
        p('ground_truth_topic', '/sauvc_auv/odometry'); p('ground_truth_frame', 'ned')

        g = lambda n: self.get_parameter(n).value
        self.rate_hz = g('rate_hz')
        self.dt = 1.0 / self.rate_hz
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

        # --- hold controller ---
        self.hold_enabled = bool(g('hold_enabled'))
        self.gains = self._gains_from_params()
        self.hold = HoldController(self.gains)
        self.feedback_timeout = g('feedback_timeout')
        self.fb = None                  # latest Feedback (ENU) or None
        self.fb_stamp = 0.0             # wall time of last feedback

        self.pub = self.create_publisher(Twist, '/cmd/setpoint', 10)
        self.create_subscription(PoseWithCovarianceStamped, '/depth', self.on_depth, 10)

        # feedback subscription is (re)created dynamically so the source can change live
        self.fb_source = g('feedback_source')
        if self.fb_source not in _FEEDBACK_SOURCES:
            raise ValueError(f"feedback_source must be one of {_FEEDBACK_SOURCES}, "
                             f"got {self.fb_source!r}")
        self._fb_sub = None
        self._fb_frame = 'enu'
        self._resubscribe_feedback(self.fb_source)

        # runtime tuning: steps, gains, hold toggle, and feedback source
        self.add_on_set_parameters_callback(self._on_set_params)

        self.get_logger().info(
            f"keyboard_teleop up, depth_mode='{self.mode}', "
            f"hold={'on' if (self.hold_enabled and self.mode=='absolute') else 'off'}, "
            f"feedback='{self.fb_source}'. Waiting for /depth to seed the initial hold "
            "target before publishing (avoids a lurch)...")

        print(HELP_TEMPLATE.format(mode=self.mode, hold=self._hold_status_word(),
                                   fb=self.fb_source))

    # ---- param helpers ----
    def _gains_from_params(self) -> HoldGains:
        g = lambda n: self.get_parameter(n).value
        return HoldGains(
            yaw_kp=g('hold_yaw_kp'), yaw_ki=g('hold_yaw_ki'), yaw_kd=g('hold_yaw_kd'),
            yaw_rate_limit=g('max_yaw'),
            pos_kp=g('hold_pos_kp'), pos_ki=g('hold_pos_ki'), pos_kd=g('hold_pos_kd'),
            vel_limit=min(g('max_surge'), g('max_sway')),
            engage_deadband=g('hold_engage_deadband'))

    def _hold_status_word(self):
        if self.mode != 'absolute':
            return "off (pulse: ArduSub holds)"
        return "on" if self.hold_enabled else "off"

    # ---- feedback wiring ----
    def _topic_and_frame(self, source):
        if source == 'none':
            return None, None
        topic = self.get_parameter(f'{source}_topic').value
        frame = self.get_parameter(f'{source}_frame').value
        if frame not in ('enu', 'ned'):
            raise ValueError(f"{source}_frame must be 'enu' or 'ned', got {frame!r}")
        return topic, frame

    def _resubscribe_feedback(self, source):
        if self._fb_sub is not None:
            self.destroy_subscription(self._fb_sub)
            self._fb_sub = None
        self.fb = None                      # drop stale pose from the old source
        self.fb_source = source
        topic, frame = self._topic_and_frame(source)
        self._fb_frame = frame
        if topic is None:
            self.get_logger().info("feedback_source='none' -> hold disabled (open-loop).")
            return
        self._fb_sub = self.create_subscription(Odometry, topic, self.on_feedback, 10)
        self.get_logger().info(f"feedback '{source}' <- {topic} ({frame}, converted to ENU)")

    def on_feedback(self, msg: Odometry):
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        if self._fb_frame == 'ned':
            ex, ey, _ = ned_to_enu_vec((pos.x, pos.y, pos.z))
            qe = ned_frd_quat_to_enu_flu((q.x, q.y, q.z, q.w))
            yaw = _yaw_from_quat_xyzw(qe[0], qe[1], qe[2], qe[3])
        else:  # 'enu' already
            ex, ey = pos.x, pos.y
            yaw = _yaw_from_quat_xyzw(q.x, q.y, q.z, q.w)
        self.fb = Feedback(x=ex, y=ey, yaw=yaw)
        self.fb_stamp = time.time()

    def _feedback_fresh(self):
        return (self.fb is not None
                and (time.time() - self.fb_stamp) < self.feedback_timeout)

    # ---- depth seeding ----
    def on_depth(self, msg):
        d = -msg.pose.pose.position.z
        was_unset = self.state.depth_target is None
        seed_depth(self.state, d)
        if was_unset and self.state.depth_target is not None:
            self.get_logger().info(f'seeded depth target = {d:.2f} m from /depth')

    # ---- runtime parameter updates ----
    def _on_set_params(self, params):
        overrides = {prm.name: prm.value for prm in params}

        # Validate first so a bad value rejects the whole set atomically.
        src = overrides.get('feedback_source')
        if src is not None and src not in _FEEDBACK_SOURCES:
            return SetParametersResult(
                successful=False,
                reason=f"feedback_source must be one of {_FEEDBACK_SOURCES}")

        # Simple scalar live-updates.
        if 'surge_step' in overrides:      self.state.surge_step = float(overrides['surge_step'])
        if 'sway_step' in overrides:       self.state.sway_step = float(overrides['sway_step'])
        if 'yaw_step' in overrides:        self.state.yaw_step = float(overrides['yaw_step'])
        if 'depth_step' in overrides:      self.state.depth_step = float(overrides['depth_step'])
        if 'hold_enabled' in overrides:    self.hold_enabled = bool(overrides['hold_enabled'])
        if 'feedback_timeout' in overrides: self.feedback_timeout = float(overrides['feedback_timeout'])

        # Gains depend on several params at once; recompute wholesale. get_parameter()
        # still returns OLD values inside this callback, so fold in the pending overrides.
        if any(k.startswith('hold_yaw_') or k.startswith('hold_pos_')
               or k == 'hold_engage_deadband' for k in overrides):
            self.gains = self._gains_with_overrides(overrides)
            self.hold.set_gains(self.gains)

        if src is not None:
            self._resubscribe_feedback(src)

        return SetParametersResult(successful=True)

    def _gains_with_overrides(self, ov):
        g = lambda n: ov[n] if n in ov else self.get_parameter(n).value
        return HoldGains(
            yaw_kp=g('hold_yaw_kp'), yaw_ki=g('hold_yaw_ki'), yaw_kd=g('hold_yaw_kd'),
            yaw_rate_limit=self.get_parameter('max_yaw').value,
            pos_kp=g('hold_pos_kp'), pos_ki=g('hold_pos_ki'), pos_kd=g('hold_pos_kd'),
            vel_limit=min(self.get_parameter('max_surge').value,
                          self.get_parameter('max_sway').value),
            engage_deadband=g('hold_engage_deadband'))

    # ---- status + publish ----
    def status_line(self):
        dt = self.state.depth_target
        dt_s = f'{dt:5.2f}' if dt is not None else ' n/a '
        hold = ''
        if self.mode == 'absolute' and self.hold_enabled:
            hold = ' HOLD' if self._feedback_fresh() else ' hold(no-fb)'
        return (f'\rsurge {self.state.surge:+.2f}  sway {self.state.sway:+.2f}  '
               f'yaw {self.state.yaw:+.2f}  depth_target {dt_s}  '
               f'steps(v={self.state.surge_step:.2f} d={self.state.depth_step:.2f})'
               f'{hold}  ')

    def publish(self):
        # In 'absolute' mode, don't publish anything until seeded — publishing z=0.0
        # before we know the real depth would command a lurch toward the surface.
        if self.mode == 'absolute' and self.state.depth_target is None:
            return

        vx, vy, vw, z = command_twist(self.state, self.mode, time.time(), self.pulse_rate)

        # Closed-loop hold: absolute mode (Path A) only, and only with fresh feedback.
        # In pulse mode ArduSub does its own hold, so we never touch vx/vy/vw there.
        if self.mode == 'absolute' and self.hold_enabled and self._feedback_fresh():
            vx, vy, vw = self.hold.compute(vx, vy, vw, self.fb, self.dt)

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
            rclpy.spin_once(node, timeout_sec=0.0)   # flush /depth + feedback subs
            node.publish()
    except KeyboardInterrupt:
        pass
    finally:
        # Always leave the vehicle in neutral and the terminal usable. Disable hold on the
        # way out so the neutral publish is a true zero, not a hold correction.
        node.hold_enabled = False
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
