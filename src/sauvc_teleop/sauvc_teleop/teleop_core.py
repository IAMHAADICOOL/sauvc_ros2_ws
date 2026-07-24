#!/usr/bin/env python3
"""teleop_core.py — pure keyboard-teleop state machine + hold controller. No ROS, no termios.

Isolated so the key -> command transition logic is unit-testable without a real TTY
(see test/test_teleop_core.py). keyboard_teleop_node.py is a thin ROS + termios wrapper
around this.

ONLY FOUR DOF: surge, sway, yaw, and depth (heave via a depth setpoint/rate). Roll and
pitch are not represented here at all -- control_core.ThrusterMixer explicitly controls
only [Fx, Fy, Fz, Mz] and leaves roll/pitch to the vehicle's passive stability. There is
no key for them because there is nothing for a key to command.

FRAME: everything here is body FLU (REP-103), matching /cmd/setpoint's convention
(mission_node, direct_control_node, ardusub_setpoint_node all agree on this) --
  x forward, y LEFT, z up.
So: 'a' (strafe left) is +y, 'd' (strafe right) is -y. Yaw: +rate is CCW from above,
which swings the nose toward +y (left), so 'q' (turn left) is +yaw, 'e' (turn right) is
-yaw. Depth uses the pipeline's own convention instead (positive = deeper/descend,
matching /depth, direct_control_node, and ardusub_setpoint_node) rather than FLU's +z=up,
because "depth" everywhere else in this codebase is a positive-down scalar, not a z
coordinate -- keeping the same sign here avoids a second, contradictory convention.

TWO DEPTH MODES, matched to the two control paths
--------------------------------------------------
  'absolute' (Path A, direct_control_node with cmd_z_is_depth:=true)
      r/f nudge a STORED depth target; command_twist() republishes it UNCHANGED every
      tick. direct_control_node's own depth PID does the actual holding -- this module
      only remembers where you left the target. This is the literal implementation of
      "stay where I leave the depth key".

  'pulse' (Path B, ardusub_setpoint_node + ArduSub SITL in ALT_HOLD)
      r/f send a brief deflection (pulse_duration) then auto-return to neutral (z=0 on
      the wire). ArduSub's own ALT_HOLD controller holds depth once z returns to
      neutral -- this module does NOT hold depth in this mode, it only nudges. This
      mirrors how a safety pilot actually flies ArduSub: tap, release, the autopilot
      holds.

PLANAR + HEADING HOLD (the HoldController below)
------------------------------------------------
direct_control_node closes VELOCITY loops on surge/sway/yaw: when you command zero it
only drives *velocity* to zero, it does not pin *position* or *heading*. So the moment
you let go the vehicle keeps whatever momentum it had, then wanders off with waves/
currents. HoldController is the missing OUTER loop: when an axis command is ~0 it captures
the current pose from a feedback estimate and runs a position/heading PID whose output is
a *velocity* command -- exactly what direct_control_node's inner velocity loop already
consumes. So it's a clean cascade (position error -> desired velocity -> thrust) and
direct_control_node stays byte-identical.

  * yaw hold engages whenever the yaw command is ~0 (independent of translation), so the
    heading stays put while you translate straight.
  * translation hold is PER-AXIS (body forward / body left), decoupled. A world "anchor"
    is slid along whichever body axis you are actively driving, so only a RELEASED axis's
    offset is enforced:
      - drive surge only  -> forward slides, LEFT offset held  => dead-straight forward
      - drive sway only   -> left slides, FORWARD offset held  => pure strafe, no creep
      - drive both        -> both slide, nothing held          => free diagonal
      - release ONE       -> that axis freezes at its current line and its PID cancels the
                             leftover momentum, so a released-sway diagonal collapses to a
                             straight surge (and vice-versa) instead of coasting on
      - release BOTH      -> both held => full station-keep against waves/current
    Heading hold keeps forward/left fixed in the world during a translate, so these
    projections stay stable.

HoldController is ONLY for 'absolute' mode (Path A). In 'pulse' mode ArduSub runs its own
loiter/position + depth hold; layering a second hold on top would fight the autopilot, so
the node must not call HoldController when mode == 'pulse'.

FEEDBACK FRAME: HoldController works entirely in ENU/FLU. The node is responsible for
converting whatever estimator it's subscribed to (some are NED) into ENU *before* handing
a Feedback in, using the workspace's one sanctioned conversion (sauvc_sim_bridge.frames).
This module never sees NED.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

STEP_SCALE_MIN = 0.02
STEP_SCALE_MAX = 2.0
STEP_SCALE_FACTOR = 1.3


@dataclass
class TeleopLimits:
    max_surge: float = 0.6
    max_sway: float = 0.6
    max_yaw: float = 0.6
    min_depth: float = 0.0
    max_depth: float = 1.5


@dataclass
class TeleopState:
    surge: float = 0.0
    sway: float = 0.0
    yaw: float = 0.0
    depth_target: Optional[float] = None   # None until seeded from a real /depth reading
    surge_step: float = 0.15
    sway_step: float = 0.15
    yaw_step: float = 0.10                  # lowered default; per-keystroke yaw increment
    depth_step: float = 0.1
    pulse_until: float = 0.0               # wall time; 'pulse' mode only
    pulse_sign: float = 0.0


def seed_depth(state: TeleopState, measured_depth: float) -> TeleopState:
    """Set depth_target from the FIRST real /depth reading only. Never overwrites a
    target the user has already started adjusting -- that would fight manual control
    with whatever the vehicle happens to be doing (e.g. mid-descent from spawn)."""
    if state.depth_target is None:
        state.depth_target = measured_depth
    return state


def apply_key(state: TeleopState, key: str, limits: TeleopLimits,
              mode: str, now: float, pulse_duration: float) -> TeleopState:
    """Pure state transition for one keypress. Mutates and returns state. Unknown keys
    are a no-op (so garbage/modifier bytes from the terminal don't raise).

    NOTE on the per-keystroke increment (request #1): the amount of command generated by
    ONE keypress is state.surge_step / state.sway_step / state.yaw_step. Those are plain
    parameters -- set them lower for finer control. The +/- and [ / ] keys still scale
    them live, and the node also lets you `ros2 param set` them at runtime.
    """
    if key == 'w':
        state.surge = min(limits.max_surge, state.surge + state.surge_step)
    elif key == 's':
        state.surge = max(-limits.max_surge, state.surge - state.surge_step)
    elif key == 'a':                                    # strafe LEFT = +y (FLU)
        state.sway = min(limits.max_sway, state.sway + state.sway_step)
    elif key == 'd':                                    # strafe RIGHT = -y (FLU)
        state.sway = max(-limits.max_sway, state.sway - state.sway_step)
    elif key == 'q':                                    # turn LEFT / CCW = +yaw
        state.yaw = min(limits.max_yaw, state.yaw + state.yaw_step)
    elif key == 'e':                                    # turn RIGHT / CW = -yaw
        state.yaw = max(-limits.max_yaw, state.yaw - state.yaw_step)
    elif key == ' ':
        state.surge = state.sway = state.yaw = 0.0
    elif key == 'x':
        state.surge = state.sway = state.yaw = 0.0
        if mode == 'pulse':
            state.pulse_until = 0.0            # cancel any in-flight pulse too
        # 'absolute' mode: depth_target is untouched -> hold continues where it was.
    elif key in ('r', 'f'):
        sign = -1.0 if key == 'r' else 1.0     # r = shallower(up) = -depth, f = deeper = +depth
        if mode == 'absolute':
            if state.depth_target is not None:
                state.depth_target += sign * state.depth_step
                state.depth_target = max(limits.min_depth,
                                         min(limits.max_depth, state.depth_target))
        else:  # 'pulse'
            state.pulse_until = now + pulse_duration
            state.pulse_sign = sign
    elif key == '0':
        if mode == 'absolute' and state.depth_target is not None:
            state.depth_target = 0.0
        # 'pulse' mode has no absolute target to reset -- '0' is a no-op there.
    elif key == '+':
        state.surge_step = min(STEP_SCALE_MAX, state.surge_step * STEP_SCALE_FACTOR)
        state.sway_step = min(STEP_SCALE_MAX, state.sway_step * STEP_SCALE_FACTOR)
    elif key == '-':
        state.surge_step = max(STEP_SCALE_MIN, state.surge_step / STEP_SCALE_FACTOR)
        state.sway_step = max(STEP_SCALE_MIN, state.sway_step / STEP_SCALE_FACTOR)
    elif key == ']':
        state.depth_step = min(STEP_SCALE_MAX, state.depth_step * STEP_SCALE_FACTOR)
    elif key == '[':
        state.depth_step = max(STEP_SCALE_MIN, state.depth_step / STEP_SCALE_FACTOR)
    return state


def command_twist(state: TeleopState, mode: str, now: float, pulse_rate: float):
    """Return (surge, sway, yaw, z) to publish RIGHT NOW, given current state and mode.

    Called every tick regardless of whether a key was just pressed -- that repetition
    is what makes 'absolute' mode a hold (same target published forever) and 'pulse'
    mode a timed nudge (nonzero only inside the pulse window, then exactly 0.0).

    This is the OPEN-LOOP command: surge/sway/yaw are the raw operator setpoints. When
    hold is enabled (absolute mode only), the node feeds these three through
    HoldController.compute(), which overrides any axis whose command is ~0 with a
    closed-loop hold. z is unaffected either way -- depth already holds via depth_target.
    """
    if mode == 'absolute':
        z = 0.0 if state.depth_target is None else state.depth_target
    else:
        z = state.pulse_sign * pulse_rate if now < state.pulse_until else 0.0
    return state.surge, state.sway, state.yaw, z


# ===========================================================================
# Closed-loop hold (requests #2, #3, #4) -- pure, ENU/FLU only.
# ===========================================================================

def wrap_pi(a: float) -> float:
    """Wrap an angle to (-pi, pi]. Used so a heading error near +/-180 deg takes the
    short way round instead of spinning the long way."""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


@dataclass
class Feedback:
    """One pose estimate, already converted to ENU/FLU by the node.

    x, y : world planar position [m] (whatever the estimator's world frame is, ENU).
    yaw  : heading [rad], CCW from world +x (ENU convention, matches the +yaw=left rule).
    """
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0


@dataclass
class HoldGains:
    """Outer-loop gains. Position PIDs turn a position error [m] into a body velocity
    command [m/s]; the heading PID turns a heading error [rad] into a yaw-rate command
    [rad/s]. Both outputs are what direct_control_node's inner velocity loops already
    expect, so these live one level up from the controller gains in control_direct.yaml.

    Defaults are deliberately gentle (the vehicle is small and the thrusters strong).
    Tune yaw first, then planar. kd defaults to 0: derivative-on-measurement of a
    wrapping heading is spiky, and position hold rarely needs D at these speeds.

    engage_deadband: |command| below this counts as "not asking to move on this axis",
    so hold takes over. Operator zeroing (space / x) sets exactly 0.0, so any small
    positive value works; it mainly guards against float dust.
    """
    yaw_kp: float = 1.2
    yaw_ki: float = 0.0
    yaw_kd: float = 0.0
    yaw_rate_limit: float = 0.6      # clamp on the yaw-rate command [rad/s]

    pos_kp: float = 0.8
    pos_ki: float = 0.0
    pos_kd: float = 0.0
    vel_limit: float = 0.4           # clamp on each body velocity command [m/s]

    engage_deadband: float = 1e-3


@dataclass
class _PID:
    """Compact single-axis PID mirroring sauvc_sim_bridge.control_core.PID conventions
    (derivative-on-measurement, conditional anti-windup, output clamp) but taking the
    error DIRECTLY so the caller can pre-wrap a heading error. Kept internal to
    teleop_core so this module has zero cross-package deps and stays trivially testable.
    """
    kp: float
    ki: float
    kd: float
    out_limit: float
    i_limit: Optional[float] = None
    integral: float = field(default=0.0, init=False)
    prev_meas: Optional[float] = field(default=None, init=False)

    def __post_init__(self):
        if self.i_limit is None:
            self.i_limit = self.out_limit

    def reset(self):
        self.integral = 0.0
        self.prev_meas = None

    def step(self, err: float, meas: float, dt: float) -> float:
        if dt <= 0.0:
            return 0.0
        d_meas = 0.0 if self.prev_meas is None else (meas - self.prev_meas) / dt
        self.prev_meas = meas

        p = self.kp * err
        d = -self.kd * d_meas
        raw = p + self.ki * self.integral + d

        saturated = abs(raw) >= self.out_limit
        pushing_out = (raw > 0 and err > 0) or (raw < 0 and err < 0)
        if not (saturated and pushing_out):
            self.integral += err * dt
            self.integral = max(-self.i_limit, min(self.i_limit, self.integral))

        out = p + self.ki * self.integral + d
        return max(-self.out_limit, min(self.out_limit, out))


class HoldController:
    """Outer position/heading hold. See module docstring for the cascade rationale.

    Usage per tick (node side, absolute mode, fresh feedback only):
        surge, sway, yaw = hold.compute(user_surge, user_sway, user_yaw, fb, dt)
    where user_* are the raw operator setpoints from command_twist(). Any axis whose
    command is ~0 comes back as a closed-loop hold value; any axis you're actively
    driving is passed through untouched.
    """

    def __init__(self, gains: HoldGains):
        self.set_gains(gains)
        # Latched targets (ENU). None until first engaged.
        self.hold_yaw: Optional[float] = None
        self.hold_x: Optional[float] = None      # world anchor, slid per-axis (see compute)
        self.hold_y: Optional[float] = None
        self._yaw_engaged = False
        self._fwd_engaged = False                # body-forward (surge) hold latched?
        self._left_engaged = False               # body-left (sway) hold latched?

    def set_gains(self, gains: HoldGains):
        """(Re)build the PIDs from gains. Called at init and whenever a gain param changes
        at runtime. Preserves latched targets so live-tuning doesn't jump the hold point."""
        self.gains = gains
        self._yaw_pid = _PID(gains.yaw_kp, gains.yaw_ki, gains.yaw_kd, gains.yaw_rate_limit)
        # One PID per BODY axis (forward / left), not per world axis: the hold error is
        # projected into the body frame so surge and sway can be held independently.
        self._fwd_pid = _PID(gains.pos_kp, gains.pos_ki, gains.pos_kd, gains.vel_limit)
        self._left_pid = _PID(gains.pos_kp, gains.pos_ki, gains.pos_kd, gains.vel_limit)

    def compute(self, user_surge: float, user_sway: float, user_yaw: float,
                fb: Feedback, dt: float):
        """Return (surge, sway, yaw) after applying hold on any ~0 axis. fb must be
        ENU/FLU and reasonably fresh (the node guards freshness)."""
        g = self.gains

        # --- heading hold (engages on yaw~0, independent of translation) ---
        if abs(user_yaw) > g.engage_deadband:
            self.hold_yaw = fb.yaw          # track heading while actively yawing
            self._yaw_pid.reset()
            self._yaw_engaged = False
            yaw_out = user_yaw
        else:
            if not self._yaw_engaged or self.hold_yaw is None:
                self.hold_yaw = fb.yaw      # latch on release
                self._yaw_pid.reset()
                self._yaw_engaged = True
            err = wrap_pi(self.hold_yaw - fb.yaw)
            yaw_out = self._yaw_pid.step(err, fb.yaw, dt)

        # --- per-axis translation hold (body forward / body left, decoupled) --------
        # A single world anchor A=(hold_x,hold_y) is SLID every tick along whichever body
        # axis you are actively driving, so only a RELEASED axis's offset is enforced.
        # Because heading hold (above) pins fwd/left in the world during a translate, the
        # forward/left projections are stable, and the two axes decouple cleanly:
        #   surge only -> forward slides, left held  => straight line, zero side-creep
        #   sway  only -> left slides, forward held  => pure strafe
        #   both       -> both slide                 => free diagonal
        #   drop sway  -> left freezes at the current line; its PID kills the diagonal
        #                 momentum so you continue straight ahead (and the mirror case)
        #   drop both  -> both held                  => full station-keep
        if self.hold_x is None:
            self.hold_x, self.hold_y = fb.x, fb.y

        c, s = math.cos(fb.yaw), math.sin(fb.yaw)     # world: fwd=(c,s), left=(-s,c)
        ex, ey = self.hold_x - fb.x, self.hold_y - fb.y
        e_fwd = c * ex + s * ey                        # anchor-minus-pos along body forward
        e_left = -s * ex + c * ey                      # ...along body left (+y, FLU)
        p_fwd = c * fb.x + s * fb.y                     # measured forward pos (PID D term)
        p_left = -s * fb.x + c * fb.y                   # measured left pos
        db = g.engage_deadband

        # forward (surge) axis
        if abs(user_surge) > db:
            self.hold_x -= e_fwd * c                    # slide anchor to zero forward error
            self.hold_y -= e_fwd * s
            self._fwd_pid.reset()
            self._fwd_engaged = False
            surge_out = user_surge
        else:
            if not self._fwd_engaged:                   # just released -> latch & clean D/I
                self._fwd_pid.reset()
                self._fwd_engaged = True
            surge_out = self._fwd_pid.step(e_fwd, p_fwd, dt)

        # left (sway) axis. Sliding along fwd above leaves e_left unchanged (fwd _|_ left),
        # so the value computed before the forward slide is still exact here.
        if abs(user_sway) > db:
            self.hold_x += e_left * s                   # slide anchor to zero lateral error
            self.hold_y -= e_left * c
            self._left_pid.reset()
            self._left_engaged = False
            sway_out = user_sway
        else:
            if not self._left_engaged:
                self._left_pid.reset()
                self._left_engaged = True
            sway_out = self._left_pid.step(e_left, p_left, dt)

        return surge_out, sway_out, yaw_out
