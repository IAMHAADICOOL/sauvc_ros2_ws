#!/usr/bin/env python3
"""Waypoint mission for the SAUVC sim — follows (x, y, z, yaw) waypoints at a constant
depth, with a selectable trajectory and a selectable localisation source.

TWO OUTPUT MODES (choose with -p output_mode:=...)
--------------------------------------------------
  'cmd'  (DEFAULT)  -> publishes geometry_msgs/Twist on /cmd/setpoint, body FLU, exactly
                      like keyboard_teleop_node. It runs the OUTER loop only (position ->
                      body velocity, heading -> yaw rate) and hands a constant absolute
                      depth through in linear.z. direct_control_node (Path A) runs the
                      inner velocity/depth loops and drives the thrusters.

                      *** This is the mode to use alongside teleop_direct.launch.py. ***
                      That launch starts direct_control_node (which OWNS
                      /sauvc_auv/thruster_setpoints) with cmd_z_is_depth:=true, so
                      linear.z is an absolute depth setpoint [m, +down]. Publishing
                      thrusters directly here would COLLIDE with direct_control_node's
                      failsafe zero-spam and the vehicle would not move -- which is the
                      whole reason this mode exists.

  'thrusters'       -> self-contained, like depth_pid_mission: closes its own depth PID
                      and writes /sauvc_auv/thruster_setpoints via the geometric mixer.
                      Use ONLY when direct_control_node is NOT running, or nothing moves.

WHAT MUST BE RUNNING (both modes)
  sim_drivers (imu_shim/depth_shim/image_relay) must be up or there is no /imu/data,
  /depth, or /camera_down/image_raw -- which means flow_eval_node publishes no /eval/*,
  which means this node gets no position feedback and never commands anything. In 'cmd'
  mode, direct_control_node must also be running. teleop_direct.launch.py provides both.
  See the chat message for the exact terminal sequence.

TRAJECTORIES (choose with -p trajectory:=N; extend by adding to TRAJECTORIES below)
  1  SQUARE      four legs; at each corner ROTATE 90 deg in place (slowly), then drive
                 forward -- noses into the direction of travel instead of swaying.
  2  FWD_CIRCLE  drive forward `fwd_distance`, then trace a full circle, nose tangent.
  3  COVERAGE    random-but-covering sweep of the pool interior, skipping the bins /
                 flares / gate / target-wall drums; yaw randomised at every waypoint so
                 the vehicle rotates constantly and moves fwd/back/sideways.

FEEDBACK SOURCE (choose with -p feedback_source:=NAME)
  Position (x, y) from ONE /eval/<name> (nav_msgs/Odometry):
      ground_truth | flow | ekf | gtsam | tile_grid   (also: pressure)
  Those messages carry no orientation, so HEADING is taken from /imu/data -- which is why
  turns are kept slow (low max yaw rate): slow yaw keeps the gyro-integrated heading from
  drifting, so IMU error stays small.

FRAMES  Work frame is NED; /cmd/setpoint output is body FLU (x fwd, y LEFT, +yaw = CCW),
  matching teleop_core / direct_control_node. /eval/* defaults to NED (set
  -p source_frame:=enu if you launched flow_eval with compare_frame:=enu). Heading
  yaw_ned = pi/2 - yaw_enu from the ENU/FLU /imu/data quaternion, as in flow_eval_node.

  ros2 run sauvc_motion_demo waypoint_pid_mission --ros-args \
      -p trajectory:=2 -p feedback_source:=ekf -p target_depth:=0.8
"""
import math
import random

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Twist, PoseWithCovarianceStamped
from std_msgs.msg import Float64MultiArray
from nav_msgs.msg import Odometry

# Mixer + PID are only needed for 'thrusters' mode; import lazily-tolerant so 'cmd' mode
# runs even if sauvc_sim_bridge isn't on the path.
try:
    from sauvc_sim_bridge.control_core import ThrusterMixer, PID
    _HAVE_MIXER = True
except Exception:                                            # pragma: no cover
    _HAVE_MIXER = False


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def yaw_enu_from_quat(x, y, z, w):
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class HeadingPID:
    """PID on heading error -> a NED yaw-RATE command (+ = turn CW / raise heading).

    Derivative is taken on the (continuous, unwrapped) measured heading, not on the
    error, so switching the leg's target yaw does not kick the derivative. The integral
    removes steady-state heading offset, which is what lets the vehicle sit on e.g. 90
    deg exactly while it translates. Anti-windup freezes the integrator when the output
    is already saturated and the error would push it further out.
    """

    def __init__(self, kp, ki, kd, out_limit, i_limit):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_limit, self.i_limit = out_limit, i_limit
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_psi = None

    def rate_cmd(self, err, psi, dt):
        d_psi = 0.0 if self.prev_psi is None else (psi - self.prev_psi) / dt
        self.prev_psi = psi
        raw = self.kp * err + self.ki * self.integral - self.kd * d_psi
        saturated = abs(raw) >= self.out_limit
        pushing_out = (raw > 0 and err > 0) or (raw < 0 and err < 0)
        if not (saturated and pushing_out):
            self.integral += err * dt
            self.integral = max(-self.i_limit, min(self.i_limit, self.integral))
        out = self.kp * err + self.ki * self.integral - self.kd * d_psi
        return max(-self.out_limit, min(self.out_limit, out))


# ---------------------------------------------------------------------------------------
# Trajectory builders. Each returns [(x, y, yaw), ...] in the NED work frame; depth is
# applied separately (constant). start = (x0, y0, yaw0) captured at run time.
# ---------------------------------------------------------------------------------------
def _unit(h):
    return np.array([math.cos(h), math.sin(h)])          # NED forward for heading h


def build_square(start, cfg):
    x0, y0, yaw0 = start
    side = cfg.p('square_side')
    turn = cfg.p('turn_dir') * math.pi / 2.0
    pos = np.array([x0, y0]); h = yaw0
    wps = []
    for _ in range(4):                       # 4 corners only; no in-place duplicates.
        pos = pos + side * _unit(h)          # _legify sets each heading = leg bearing,
        wps.append([pos[0], pos[1], h])      # so turn-in-place happens automatically.
        h = wrap(h + turn)
    # The last two corners are the return legs that run back alongside the start wall;
    # pin their x to the start x so they never creep past the start line into the wall.
    wps[-2][0] = x0
    wps[-1][0] = x0
    return [tuple(w) for w in wps]


def build_fwd_circle(start, cfg):
    x0, y0, yaw0 = start
    fwd = cfg.p('fwd_distance'); R = cfg.p('circle_radius')
    n = int(cfg.p('circle_points')); d = cfg.p('turn_dir')
    entry = np.array([x0, y0]) + fwd * _unit(yaw0)
    wps = [(entry[0], entry[1], yaw0)]
    center = entry + R * _unit(yaw0 + d * math.pi / 2.0)
    a0 = math.atan2(entry[1] - center[1], entry[0] - center[0])
    for k in range(1, n + 1):
        ang = a0 - d * 2.0 * math.pi * k / n
        pos = center + R * np.array([math.cos(ang), math.sin(ang)])
        wps.append((pos[0], pos[1], wrap(ang - d * math.pi / 2.0)))
    return wps


# Keep-out discs (NED x, y, radius[m]) over the finals bins/flares/gate/target drums.
_KEEPOUTS = [
    (4.4, -1.5, 1.3), (-6.5, -1.0, 0.9), (-1.0, 3.0, 0.9), (1.5, 4.5, 0.9),
    (0.0, -4.0, 0.9), (11.0, 0.0, 4.5), (-11.6, -2.0, 1.2),
]


def build_coverage(start, cfg):
    xmin, xmax = cfg.p('cover_x_min'), cfg.p('cover_x_max')
    ymin, ymax = cfg.p('cover_y_min'), cfg.p('cover_y_max')
    step = cfg.p('cover_step'); scale = cfg.p('keepout_scale'); veh = cfg.p('vehicle_margin')
    rng = random.Random(int(cfg.p('cover_seed')))
    keeps = [(cx, cy, r * scale + veh) for (cx, cy, r) in _KEEPOUTS]

    def blocked(px, py):
        return any((px - cx) ** 2 + (py - cy) ** 2 < r * r for (cx, cy, r) in keeps)

    cells, nx, ny = [], max(1, int(round((xmax - xmin) / step))), max(1, int(round((ymax - ymin) / step)))
    for i in range(nx + 1):
        for j in range(ny + 1):
            px = xmin + i * (xmax - xmin) / nx
            py = ymin + j * (ymax - ymin) / ny
            if not blocked(px, py):
                cells.append((px, py))
    x0, y0, _ = start
    cells.sort(key=lambda c: (c[0] - x0) ** 2 + (c[1] - y0) ** 2)
    first, rest = cells[0], cells[1:]
    rng.shuffle(rest)
    return [(px, py, wrap(rng.uniform(-math.pi, math.pi))) for (px, py) in [first] + rest]


TRAJECTORIES = {
    1: ('SQUARE', build_square),
    2: ('FWD_CIRCLE', build_fwd_circle),
    3: ('COVERAGE', build_coverage),
}


class WaypointMission(Node):
    def __init__(self):
        super().__init__('waypoint_mission')
        d = self.declare_parameter
        d('robot', 'sauvc_auv')
        d('output_mode', 'cmd')                # 'cmd' (default) | 'thrusters'
        d('trajectory', 1)
        d('feedback_source', 'ground_truth')   # ground_truth|flow|ekf|gtsam|tile_grid|pressure
        d('source_frame', 'ned')               # frame /eval/* was published in
        d('eval_topic', '')                    # optional override of /eval/<source>

        d('target_depth', 0.8)                 # [m] +down, held for every trajectory
        d('rate_hz', 20.0)
        d('pos_tol', 0.35)                     # [m]
        d('yaw_tol', 0.15)                     # [rad]
        d('wp_timeout', 30.0)                  # [s]
        # DESCEND -> RUN transition. The depth loop may settle a little short of the
        # setpoint (buoyancy + a capped integrator), so we advance when the DIVE HAS
        # SETTLED -- close to target OR no longer moving vertically -- not when depth
        # exactly equals target. descend_timeout is a hard backstop so it can never hang.
        d('descend_settle', 2.0)               # [s] settled condition must hold this long
        d('descend_tol', 0.15)                 # [m] "close enough" band around target
        d('descend_rate_tol', 0.03)            # [m/s] below this = vertical motion stopped
        d('descend_timeout', 25.0)             # [s] proceed regardless after this

        # 'cmd' mode: outer loop. position -> body-velocity setpoint (capped at the SAME
        # magnitude a single teleop surge/sway step commands, so acceleration matches
        # "give 0.15 in teleop"); heading -> yaw-rate setpoint via a real PID that HOLDS
        # the leg heading (integral kills steady-state drift, so it tracks e.g. 90 deg
        # exactly while translating).
        d('pos_kp', 0.35)                      # forward velocity setpoint per metre of error
        d('max_speed', 0.15)                   # == one teleop step; caps FORWARD command
        d('yaw_kp', 0.8)                        # heading PID
        d('yaw_ki', 0.25)
        d('yaw_kd', 0.10)
        d('max_yaw_rate', 0.15)                # == one teleop yaw step; caps turn command
        d('yaw_i_limit', 0.25)                 # anti-windup clamp on the yaw integrator

        # straight-line motion: at each waypoint STOP, TURN in place to the exact leg
        # heading, then DRIVE FORWARD ONLY (never sway) holding heading with the yaw PID.
        d('straight_line', True)               # False -> old holonomic surge+sway+yaw
        d('turn_settle', 0.4)                  # [s] heading must be held before driving
        d('turn_rate_tol', 0.06)               # [rad/s] below this = finished turning
        d('turn_timeout', 40.0)                # [s]
        d('brake_settle', 0.6)                 # [s] must be stopped this long at a waypoint
        d('stop_speed_tol', 0.03)              # [m/s] below this = stopped translating
        d('brake_timeout', 6.0)                # [s]

        # 'thrusters' mode: full inner loop here (only used when direct_control_node absent)
        d('surge_kp', 0.6); d('surge_ki', 0.02); d('surge_kd', 1.2); d('max_surge', 0.45)
        d('sway_kp', 0.6);  d('sway_ki', 0.02);  d('sway_kd', 1.2);  d('max_sway', 0.45)
        d('t_yaw_kp', 0.5); d('t_yaw_ki', 0.0);  d('t_yaw_kd', 0.6); d('t_max_yaw', 0.15)
        d('depth_kp', 3.0); d('depth_ki', 0.5);  d('depth_kd', 0.8); d('max_depth', 0.6)

        # trajectory shapes
        d('turn_dir', 1.0)
        d('square_side', 4.0)
        d('fwd_distance', 5.0)
        d('circle_radius', 3.0); d('circle_points', 24)
        d('cover_x_min', -10.5); d('cover_x_max', 9.5)
        d('cover_y_min', -6.5);  d('cover_y_max', 6.5)
        d('cover_step', 3.0); d('cover_seed', 1); d('keepout_scale', 1.0)
        d('vehicle_margin', 0.6)

        self.robot = self.p('robot')
        self.mode = self.p('output_mode')
        if self.mode not in ('cmd', 'thrusters'):
            raise ValueError("output_mode must be 'cmd' or 'thrusters'")
        self.source = self.p('feedback_source')
        self.src_frame = self.p('source_frame')
        topic = self.p('eval_topic') or f'/eval/{self.source}'
        self.dt = 1.0 / self.p('rate_hz')

        # feedback caches
        self.pos = None
        self.psi = None
        self._psi_last = None
        self.depth = None
        self._last_depth = None
        self._depth_rate = 0.0                 # smoothed |d(depth)/dt|

        # --- output plumbing per mode ---
        if self.mode == 'cmd':
            self.pub_cmd = self.create_publisher(Twist, '/cmd/setpoint', 10)
            self.hpid = HeadingPID(self.p('yaw_kp'), self.p('yaw_ki'), self.p('yaw_kd'),
                                   self.p('max_yaw_rate'), self.p('yaw_i_limit'))
        else:
            if not _HAVE_MIXER:
                raise RuntimeError("'thrusters' mode needs sauvc_sim_bridge.control_core")
            self.mixer = ThrusterMixer()
            self.surge = PID(self.p('surge_kp'), self.p('surge_ki'), self.p('surge_kd'), self.p('max_surge'))
            self.sway = PID(self.p('sway_kp'), self.p('sway_ki'), self.p('sway_kd'), self.p('max_sway'))
            self.yaw = PID(self.p('t_yaw_kp'), self.p('t_yaw_ki'), self.p('t_yaw_kd'), self.p('t_max_yaw'))
            self.depthpid = PID(self.p('depth_kp'), self.p('depth_ki'), self.p('depth_kd'), self.p('max_depth'))
            self.pub_thr = self.create_publisher(
                Float64MultiArray, f'/{self.robot}/thruster_setpoints', 10)

        # --- feedback subs ---
        self.create_subscription(Odometry, topic, self.on_odom, 10)
        self.create_subscription(Imu, '/imu/data', self.on_imu, qos_profile_sensor_data)
        self.create_subscription(PoseWithCovarianceStamped, '/depth', self.on_depth, 10)

        self.waypoints = None
        self.wp_i = 0
        self.state = 'WAIT'
        self.state_t0 = self.now()
        self.settled_since = None
        # straight-line RUN sub-state
        self.phase = 'TURN'                    # 'TURN' | 'DRIVE' | 'BRAKE'
        self.phase_t0 = self.now()
        self.leg_heading = 0.0
        self.phase_ok_since = None
        self.anchor = None                     # (x,y) held during TURN/BRAKE (corner)
        self._pos_prev = None
        self._speed = 0.0                      # smoothed horizontal speed [m/s]
        self._psi_prev = None
        self._yaw_rate = 0.0                   # smoothed |d(psi)/dt| [rad/s]

        traj = int(self.p('trajectory'))
        name = TRAJECTORIES.get(traj, ('?',))[0]
        self.get_logger().info(
            f"waypoint mission up [{self.mode}]: trajectory {traj} ({name}), "
            f"feedback '{self.source}' on {topic} [{self.src_frame}]. "
            f"Waiting for /eval feedback + /imu/data + /depth ...")
        self.timer = self.create_timer(self.dt, self.step)

    # -- helpers --
    def p(self, n):
        return self.get_parameter(n).value

    def now(self):
        return self.get_clock().now().nanoseconds * 1e-9

    # -- callbacks --
    def on_odom(self, msg):
        x, y = msg.pose.pose.position.x, msg.pose.pose.position.y
        if self.src_frame == 'enu':
            x, y = y, x                       # (E, N) -> (N, E)
        self.pos = (x, y)

    def on_imu(self, msg):
        q = msg.orientation
        psi = math.pi / 2.0 - yaw_enu_from_quat(q.x, q.y, q.z, q.w)   # ENU CCW -> NED CW
        if self._psi_last is None:
            self.psi = psi
        else:
            self.psi += wrap(psi - self._psi_last)                    # unwrap
        self._psi_last = psi

    def on_depth(self, msg):
        self.depth = -msg.pose.pose.position.z                        # z = -depth

    # -- body-frame errors from a NED target --
    def body_errors(self, tx, ty):
        c, s = math.cos(self.psi), math.sin(self.psi)
        e_n, e_e = tx - self.pos[0], ty - self.pos[1]
        fwd = c * e_n + s * e_e            # body forward
        left = s * e_n - c * e_e           # body LEFT (FLU +y)
        return fwd, left

    def enter(self, state):
        self.get_logger().info(f'-> {state}')
        self.state = state
        self.state_t0 = self.now()

    def _reset_yaw(self):
        if self.mode == 'cmd':
            self.hpid.reset()               # clear integrator when the leg heading changes

    def build_waypoints(self):
        traj = int(self.p('trajectory'))
        if traj not in TRAJECTORIES:
            self.get_logger().error(f'unknown trajectory {traj}; using 1'); traj = 1
        start = (self.pos[0], self.pos[1], self.psi)
        wps = TRAJECTORIES[traj][1](start, self)
        legs = self._legify(start, wps)
        self.get_logger().info(
            f'{TRAJECTORIES[traj][0]}: {len(legs)} waypoints from '
            f'({start[0]:+.2f}, {start[1]:+.2f}); leg headings '
            + ', '.join(f'{math.degrees(h):+.0f}' for _, _, h in legs))
        return legs

    def _legify(self, start, wps):
        """Assign each waypoint the EXACT heading of the straight leg leading into it:
        the NED bearing from the previous point. For the axis-aligned square this yields
        precisely 0/90/180/270; for the circle/coverage it noses toward each point. This
        is the heading the vehicle turns to and then holds while driving forward."""
        legs, prev = [], (start[0], start[1])
        for (x, y, _hint) in wps:
            h = math.atan2(y - prev[1], x - prev[0])   # NED bearing: atan2(dEast, dNorth)
            legs.append((x, y, h))
            prev = (x, y)
        return legs

    # -- motion tracking (speed + yaw rate), smoothed so noise doesn't trip the tests --
    def _update_motion(self):
        if self._pos_prev is not None:
            v = math.hypot(self.pos[0] - self._pos_prev[0],
                           self.pos[1] - self._pos_prev[1]) / self.dt
            self._speed = 0.7 * self._speed + 0.3 * v
        self._pos_prev = self.pos
        if self._psi_prev is not None:
            r = abs(self.psi - self._psi_prev) / self.dt
            self._yaw_rate = 0.7 * self._yaw_rate + 0.3 * r
        self._psi_prev = self.psi

    # -- unified output: FLU forward/left efforts + a target heading; handles both modes --
    def _emit(self, surge, left, target_heading, depth):
        yaw_err = wrap(target_heading - self.psi)
        if self.mode == 'cmd':
            ned_rate = self.hpid.rate_cmd(yaw_err, self.psi, self.dt)   # + = CW
            m = Twist()
            m.linear.x = float(surge)              # FLU forward
            m.linear.y = float(left)               # FLU LEFT (0 in straight-line mode)
            m.linear.z = float(depth)              # absolute depth (cmd_z_is_depth:=true)
            m.angular.z = float(-ned_rate)         # FLU +CCW, so negate NED CW rate
            self.pub_cmd.publish(m)
        else:
            mz_flu = self.yaw.update(self.psi + yaw_err, self.psi, self.dt)
            fz = self.depthpid.update(depth, self.depth, self.dt)
            u = self.mixer.wrench_to_thrust(surge, -left, fz, -mz_flu)  # FLU->FRD
            self.pub_thr.publish(Float64MultiArray(data=[float(v) for v in u]))

    # -- main loop --
    def step(self):
        if self.pos is None or self.psi is None or self.depth is None:
            return
        self._update_motion()
        t = self.now() - self.state_t0
        target_depth = self.p('target_depth')

        if self.state == 'WAIT':
            self.enter('DESCEND')

        if self.state == 'DESCEND':
            self._emit(0.0, 0.0, self.psi, target_depth)      # hold heading, sink
            if self._last_depth is not None:
                inst = abs(self.depth - self._last_depth) / self.dt
                self._depth_rate = 0.7 * self._depth_rate + 0.3 * inst
            self._last_depth = self.depth

            close = abs(target_depth - self.depth) < self.p('descend_tol')
            settled = (self._depth_rate < self.p('descend_rate_tol')
                       and self.depth > 0.4 * target_depth)
            if close or settled:
                if self.settled_since is None:
                    self.settled_since = self.now()
                elif self.now() - self.settled_since > self.p('descend_settle'):
                    self._begin_run()
            else:
                self.settled_since = None

            if t > self.p('descend_timeout'):
                self.get_logger().warn(
                    f'descend timeout at depth {self.depth:.2f} m (target '
                    f'{target_depth:.2f}); starting waypoints anyway')
                self._begin_run()
            self._status(0.0, 0.0); return

        if self.state == 'RUN':
            if self.p('straight_line'):
                self._run_straight(t, target_depth)
            else:
                self._run_holonomic(t, target_depth)
            return

        if self.state == 'SURFACE':
            self._emit(0.0, 0.0, self.psi, 0.05)
            if self.depth < 0.15:
                self.enter('DONE')
            self._status(0.0, 0.0); return

        if self.state == 'DONE':
            self._emit(0.0, 0.0, self.psi, 0.0)
            if t > 1.0:
                if self.mode == 'thrusters':
                    self.pub_thr.publish(Float64MultiArray(data=[0.0] * 8))
                self.get_logger().info('mission complete')
                raise SystemExit

    def _begin_run(self):
        self.settled_since = None
        self.waypoints = self.build_waypoints()
        self.wp_i = 0
        self._reset_yaw()
        self.leg_heading = self.waypoints[0][2]
        self.anchor = self.pos                 # hold the start point during the first turn
        self.phase = 'TURN'
        self.phase_t0 = self.now()
        self.phase_ok_since = None
        self.enter('RUN')

    # -- straight-line RUN: STOP -> TURN in place -> DRIVE forward only (no sway) --
    def _run_straight(self, t, depth):
        tx, ty, th = self.waypoints[self.wp_i]
        yaw_err = wrap(th - self.psi)
        dist = math.hypot(tx - self.pos[0], ty - self.pos[1])
        # remaining distance measured ALONG the leg heading (lateral offset ignored, so
        # the vehicle never strafes to chase it -- it just drives straight and stops abeam)
        along = math.cos(th) * (tx - self.pos[0]) + math.sin(th) * (ty - self.pos[1])

        if self.phase == 'TURN':
            # Hold the corner (x,y error -> 0) while ONLY the yaw changes. Momentum from
            # the previous leg would otherwise carry the vehicle out of position mid-turn;
            # station-keeping pins it so the turn is genuinely in place. This surge/sway is
            # a stationary hold, not travel, so it does not reintroduce diagonal motion.
            self._hold_position(th, depth)
            # Settled = heading within tolerance, held continuously for turn_settle. The
            # hold itself proves it isn't still swinging, so we do NOT gate on yaw RATE
            # (IMU jitter keeps a rate gate flickering and wedges the turn forever).
            done = abs(yaw_err) < self.p('yaw_tol')
            if self._hold('turn', done, self.p('turn_settle')) \
                    or (self.now() - self.phase_t0) > self.p('turn_timeout'):
                self.phase = 'DRIVE'; self.phase_t0 = self.now(); self.phase_ok_since = None

        elif self.phase == 'DRIVE':
            surge = max(0.0, min(self.p('max_speed'), self.p('pos_kp') * along))
            self._emit(surge, 0.0, th, depth)                 # forward + hold heading only
            if along < self.p('pos_tol') or dist < self.p('pos_tol') \
                    or (self.now() - self.phase_t0) > self.p('wp_timeout'):
                self.anchor = self.pos        # pin the corner we actually reached
                self.phase = 'BRAKE'; self.phase_t0 = self.now(); self.phase_ok_since = None

        else:  # BRAKE -- station-keep at the corner until fully stopped, then turn
            self._hold_position(th, depth)                    # hold corner + current heading
            stopped = self._speed < self.p('stop_speed_tol')
            if self._hold('brake', stopped, self.p('brake_settle')) \
                    or (self.now() - self.phase_t0) > self.p('brake_timeout'):
                self.wp_i += 1
                if self.wp_i >= len(self.waypoints):
                    self.enter('SURFACE')
                else:
                    self._reset_yaw()                          # new leg heading
                    self.leg_heading = self.waypoints[self.wp_i][2]
                    self.phase = 'TURN'; self.phase_t0 = self.now()
                    self.phase_ok_since = None
        self._status(along, abs(yaw_err))

    def _hold_position(self, target_heading, depth):
        """Station-keep at self.anchor (drive x,y error to zero with surge+sway) while
        commanding target_heading. Used in TURN/BRAKE so momentum can't drag the vehicle
        off the corner. Corrections are clamped to max_speed like everything else."""
        ax, ay = self.anchor if self.anchor is not None else self.pos
        c, s = math.cos(self.psi), math.sin(self.psi)
        e_n, e_e = ax - self.pos[0], ay - self.pos[1]
        fwd = c * e_n + s * e_e                                # body forward error
        left = s * e_n - c * e_e                               # body LEFT error
        k, vmax = self.p('pos_kp'), self.p('max_speed')
        surge = max(-vmax, min(vmax, k * fwd))
        left_cmd = max(-vmax, min(vmax, k * left))
        self._emit(surge, left_cmd, target_heading, depth)

    def _hold(self, _tag, condition, settle):
        """True once `condition` has held continuously for `settle` seconds."""
        if condition:
            if self.phase_ok_since is None:
                self.phase_ok_since = self.now()
            return (self.now() - self.phase_ok_since) > settle
        self.phase_ok_since = None
        return False

    # -- legacy holonomic RUN (surge+sway+yaw at once); kept behind straight_line:=false --
    def _run_holonomic(self, t, depth):
        tx, ty, tyaw = self.waypoints[self.wp_i]
        c, s = math.cos(self.psi), math.sin(self.psi)
        e_n, e_e = tx - self.pos[0], ty - self.pos[1]
        fwd, left = c * e_n + s * e_e, s * e_n - c * e_e
        k, vmax = self.p('pos_kp'), self.p('max_speed')
        self._emit(max(-vmax, min(vmax, k * fwd)),
                   max(-vmax, min(vmax, k * left)), tyaw, depth)
        dist = math.hypot(e_n, e_e)
        yaw_err = wrap(tyaw - self.psi)
        if (dist < self.p('pos_tol') and abs(yaw_err) < self.p('yaw_tol')) \
                or t > self.p('wp_timeout'):
            self.wp_i += 1; self._reset_yaw()
            if self.wp_i >= len(self.waypoints):
                self.enter('SURFACE')
            else:
                self.state_t0 = self.now()
        self._status(dist, abs(yaw_err))

    def _status(self, dist, yerr):
        wp = f'{self.wp_i}/{len(self.waypoints)}' if self.waypoints else '-'
        ph = self.phase if self.state == 'RUN' else ''
        print(f'\r[{self.state:7s}{ph:>6s}] wp {wp:>7s}  pos '
              f'({self.pos[0]:+.2f},{self.pos[1]:+.2f})  depth {self.depth:+.2f}m  '
              f'yaw {math.degrees(self.psi):+.0f}deg  along {dist:+5.2f}m  '
              f'ye {math.degrees(yerr):3.0f}deg  v {self._speed:.2f}   ',
              end='', flush=True)


def main():
    rclpy.init()
    try:
        rclpy.spin(WaypointMission())
    except (SystemExit, KeyboardInterrupt):
        pass


if __name__ == '__main__':
    main()
