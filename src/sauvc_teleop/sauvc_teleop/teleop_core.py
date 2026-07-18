#!/usr/bin/env python3
"""teleop_core.py — pure keyboard-teleop state machine. No ROS, no termios.

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
"""

from dataclasses import dataclass, field
from typing import Optional

STEP_SCALE_MIN = 0.02
STEP_SCALE_MAX = 2.0
STEP_SCALE_FACTOR = 1.3


@dataclass
class TeleopLimits:
    max_surge: float = 0.6
    max_sway: float = 0.6
    max_yaw: float = 1.2
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
    yaw_step: float = 0.3
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
    are a no-op (so garbage/modifier bytes from the terminal don't raise)."""
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
    """
    if mode == 'absolute':
        z = 0.0 if state.depth_target is None else state.depth_target
    else:
        z = state.pulse_sign * pulse_rate if now < state.pulse_until else 0.0
    return state.surge, state.sway, state.yaw, z
