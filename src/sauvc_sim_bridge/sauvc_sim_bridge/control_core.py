#!/usr/bin/env python3
"""control_core.py — pure-python control math for the direct-PID path. No ROS imports.

Two pieces, both unit-testable offline (same philosophy as flow_core.FlowVelocityEstimator):

  ThrusterMixer   : maps a desired body wrench (Fx, Fy, Fz, Mz) to 8 normalized thruster
                    setpoints, using a pseudo-inverse of the allocation matrix DERIVED
                    from the my_auv.scn geometry (not hand-guessed).
  PID             : a single-axis PID with output clamp, integral anti-windup, and
                    derivative-on-measurement (not on error, so setpoint steps don't kick).

WHY A GEOMETRIC MIXER, NOT HAND-TUNED GAINS PER THRUSTER
--------------------------------------------------------
mission_node emits /cmd/setpoint as a body-frame Twist: desired vx, vy, vz, yaw-rate. To
turn that into 8 thruster commands you need the vehicle's actuator geometry, which is
fully specified in my_auv.scn. Building the allocation matrix from those origins and
angles means the mixer stays correct if a thruster moves, and it documents WHY each
thruster fires the way it does. The alternative — eyeballing which thrusters to fire for
"turn left" — is exactly the class of error that produced the "vehicle doesn't move then
topples" bug on the sign convention.

THRUSTER ORDER (matches my_auv.scn and the ArduSub bridge and depth_pid_mission):
    [0] HFP  [1] HFS  [2] HAP  [3] HAS    horizontal, vectored 45 deg
    [4] VFP  [5] VFS  [6] VAP  [7] VAS    vertical; POSITIVE = thrust DOWN (descend)

Note the vertical sign: +Fz in body FRD is DOWN, and a positive vertical setpoint
descends. depth_pid_mission uses the same convention, so a positive heave command and a
positive depth-PID output both mean "go deeper". Keep it consistent or the depth loop
inverts.

FRAME: everything here is body FRD, matching the sim's native frame and the thruster
geometry. The mixer is downstream of all localization, so it never sees ENU — do not
convert here.
"""

import numpy as np


# Thruster geometry from my_auv.scn: (x, y, z) origin [m] and yaw [rad] for horizontals.
# Verticals point along +Z body (down). This table is the single source of truth for the
# mixer; if the scene changes, change it here.
_THRUSTERS = [
    # name,   x,      y,      z,      yaw,      kind
    ('HFP',  0.245, -0.215, -0.004,  0.7854, 'H'),
    ('HFS',  0.245,  0.215, -0.004, -0.7854, 'H'),
    ('HAP', -0.245, -0.215, -0.004, -0.7854, 'H'),
    ('HAS', -0.245,  0.215, -0.004,  0.7854, 'H'),
    ('VFP',  0.158, -0.120,  0.0125, 0.0,    'V'),
    ('VFS',  0.158,  0.120,  0.0125, 0.0,    'V'),
    ('VAP', -0.158, -0.120,  0.0125, 0.0,    'V'),
    ('VAS', -0.158,  0.120,  0.0125, 0.0,    'V'),
]


def build_allocation(thrusters=_THRUSTERS):
    """6x8 allocation matrix A: wrench = A @ thrusts, rows [Fx Fy Fz Mx My Mz], body FRD."""
    A = np.zeros((6, 8))
    for i, (_name, x, y, z, yaw, kind) in enumerate(thrusters):
        if kind == 'H':
            f = np.array([np.cos(yaw), np.sin(yaw), 0.0])
        else:
            f = np.array([0.0, 0.0, 1.0])          # vertical thrust along +Z (down)
        r = np.array([x, y, z])
        A[:3, i] = f
        A[3:, i] = np.cross(r, f)
    return A


class ThrusterMixer:
    """Maps a 4-DOF body wrench (Fx, Fy, Fz, Mz) to 8 normalized [-1, 1] setpoints.

    We actively control surge, sway, heave and yaw. Roll and pitch are left to the
    vehicle's passive stability (it is bottom-heavy and the verticals are near the roll/
    pitch axes anyway), so those two rows are dropped from the allocation before
    inverting. Controlling all six from eight thrusters is possible but couples axes you
    have no sensor to close a loop on — keep it to what the mission actually commands.
    """

    #: rows of A we control: Fx, Fy, Fz, Mz
    CONTROLLED = [0, 1, 2, 5]

    def __init__(self, thrusters=_THRUSTERS):
        A = build_allocation(thrusters)
        self.Ac = A[self.CONTROLLED]          # 4x8
        self.mix = np.linalg.pinv(self.Ac)    # 8x4, minimum-norm thrust for a wrench
        self.cond = float(np.linalg.cond(self.Ac))

    def wrench_to_thrust(self, fx, fy, fz, mz, clip=True):
        """Return 8 setpoints for the requested body wrench.

        The pseudo-inverse gives the minimum-norm solution. If any thruster saturates,
        the whole vector is scaled down uniformly (preserving DIRECTION — the vehicle
        goes where commanded, just slower) rather than clipped per-thruster (which would
        distort the wrench and could, e.g., turn a pure surge into a slow yaw).
        """
        u = self.mix @ np.array([fx, fy, fz, mz], dtype=float)
        if clip:
            peak = np.max(np.abs(u))
            if peak > 1.0:
                u = u / peak
        return u


class PID:
    """Single-axis PID: output clamp, integral anti-windup (clamped + conditional),
    derivative-on-measurement.

    Derivative-on-measurement (d/dt of the measured value, negated) rather than
    d/dt of the error avoids a derivative spike every time mission_node steps the
    setpoint. Anti-windup freezes the integrator whenever the output is saturated AND
    the error would push it further into saturation.
    """

    def __init__(self, kp, ki, kd, out_limit=1.0, i_limit=None):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.out_limit = out_limit
        self.i_limit = i_limit if i_limit is not None else out_limit
        self.reset()

    def reset(self):
        self.integral = 0.0
        self.prev_meas = None

    def update(self, setpoint, meas, dt):
        if dt <= 0.0:
            return 0.0
        err = setpoint - meas
        d_meas = 0.0 if self.prev_meas is None else (meas - self.prev_meas) / dt
        self.prev_meas = meas

        p = self.kp * err
        d = -self.kd * d_meas
        raw = p + self.ki * self.integral + d

        # Conditional integration: only accumulate if not saturating outward.
        saturated = abs(raw) >= self.out_limit
        pushing_out = (raw > 0 and err > 0) or (raw < 0 and err < 0)
        if not (saturated and pushing_out):
            self.integral += err * dt
            self.integral = max(-self.i_limit, min(self.i_limit, self.integral))

        out = p + self.ki * self.integral + d
        return max(-self.out_limit, min(self.out_limit, out))
