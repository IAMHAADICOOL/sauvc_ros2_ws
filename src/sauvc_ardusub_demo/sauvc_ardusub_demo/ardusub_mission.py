#!/usr/bin/env python3
"""ArduSub mission v5 - built on the official ArduSub pymavlink patterns
(https://www.ardusub.com/developers/pymavlink.html):

  * depth via SET_POSITION_TARGET_GLOBAL_INT: ALT_HOLD is given an explicit
    depth SETPOINT and the FIRMWARE closes the loop (no manual dive throttle)
  * confirmed mode switching (retry until heartbeat reports the mode)
  * GCS heartbeats at 1 Hz from this script (failsafe requirement)
  * MANUAL_CONTROL at a constant rate during legs (pilot-input timeout)

Tuning: TARGET_DEPTH below; firmware-side PSC_* / PILOT_SPEED_* as in v2 notes.
Diagnostics kept: telemetry-type listing, EKF-vs-truth printing, topple and
max-depth watchdogs.
"""
import math
import sys
import time
from pymavlink import mavutil

TARGET_DEPTH = 0.5     # [m] positive down; sent to firmware as alt = -depth
DEPTH_TOL = 0.10       # [m] "reached" tolerance
MAX_DEPTH = 1.15       # [m] abort guard (floor at the start zone is ~1.23 m)
DIVE_TIMEOUT = 30.0
NEUTRAL_Z = 500
FWD = 400
LEG_TIME = 5.0
HOLD_TIME = 5.0
TOPPLE_DEG = 60.0

STATE = {'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0, 'depth': float('nan'),
         'p0': float('nan'), 'last_rx': 0.0, 'last_hb': 0.0}
RHO_G = 1000.0 * 9.81
SEEN = set()
BOOT = time.time()


class MissionAbort(RuntimeError):
    pass


def poll(m):
    """Heartbeat out (1 Hz) + drain telemetry in, with watchdogs."""
    if time.time() - STATE['last_hb'] > 1.0:
        STATE['last_hb'] = time.time()
        m.mav.heartbeat_send(mavutil.mavlink.MAV_TYPE_GCS,
                             mavutil.mavlink.MAV_AUTOPILOT_INVALID, 0, 0, 0)
    while True:
        msg = m.recv_match(blocking=False)
        if msg is None:
            if STATE['last_rx'] and time.time() - STATE['last_rx'] > 3.0:
                print('  (WARNING: no telemetry for 3 s)')
                STATE['last_rx'] = time.time()
            return
        STATE['last_rx'] = time.time()
        SEEN.add(msg.get_type())
        t = msg.get_type()
        if t == 'ATTITUDE':
            STATE.update(roll=math.degrees(msg.roll),
                         pitch=math.degrees(msg.pitch),
                         yaw=math.degrees(msg.yaw))
            if abs(STATE['roll']) > TOPPLE_DEG or abs(STATE['pitch']) > TOPPLE_DEG:
                raise MissionAbort(f"TOPPLE roll={STATE['roll']:.1f} "
                                   f"pitch={STATE['pitch']:.1f}")
        elif t == 'SCALED_PRESSURE2':
            # ArduSub's water-pressure sensor: press_abs [hPa].
            # First sample at the surface = zero reference.
            # (VFR_HUD.alt is ABSOLUTE AMSL - home is ~584 m at SITL's
            # default location - so it is NOT usable as depth.)
            p = msg.press_abs * 100.0
            if STATE['p0'] != STATE['p0']:
                STATE['p0'] = p
                print(f'  surface pressure reference: {p:.0f} Pa')
            STATE['depth'] = (p - STATE['p0']) / RHO_G
        elif t == 'GLOBAL_POSITION_INT' and STATE['p0'] != STATE['p0']:
            STATE['depth'] = -msg.relative_alt / 1000.0  # fallback only
        if STATE['depth'] == STATE['depth'] and STATE['depth'] > MAX_DEPTH:
            raise MissionAbort(f"DEPTH GUARD {STATE['depth']:.2f} m")


def status(label, extra=''):
    print(f"  [{label:9s}] EKF depth={STATE['depth']:+5.2f} m  "
          f"rpy=({STATE['roll']:+5.1f},{STATE['pitch']:+5.1f},"
          f"{STATE['yaw']:+6.1f}) {extra}", flush=True)


def manual_for(m, seconds, x=0, y=0, z=NEUTRAL_Z, r=0, label=''):
    t_end = time.time() + seconds
    last = 0.0
    while time.time() < t_end:
        poll(m)
        m.mav.manual_control_send(m.target_system, x, y, z, r, 0)
        if time.time() - last > 0.5:
            last = time.time()
            status(label, f'cmd(x={x} y={y} z={z} r={r})')
        time.sleep(0.1)


def set_target_depth(m, depth_m):
    """Depth setpoint for ALT_HOLD, alt = -depth RELATIVE TO HOME.
    MUST be the RELATIVE_ALT frame: with GLOBAL_INT the alt is absolute AMSL,
    and SITL's default home is ~584 m AMSL (CMAC) - so alt=-0.5 commanded a
    dive to 584.5 m below home and the vehicle drove itself into the floor.
    Home = arming spot = the surface, so relative -depth is exactly right."""
    mask = (mavutil.mavlink.POSITION_TARGET_TYPEMASK_X_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_Y_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VX_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VY_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_VZ_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AX_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AY_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_AZ_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_IGNORE |
            mavutil.mavlink.POSITION_TARGET_TYPEMASK_YAW_RATE_IGNORE)
    m.mav.set_position_target_global_int_send(
        int(1e3 * (time.time() - BOOT)), m.target_system, m.target_component,
        mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT, mask,
        0, 0, -depth_m, 0, 0, 0, 0, 0, 0, 0, 0)


def set_mode_confirmed(m, name, tries=10):
    mode_id = m.mode_mapping()[name]
    for _ in range(tries):
        m.set_mode(mode_id)
        hb = m.recv_match(type='HEARTBEAT', blocking=True, timeout=1)
        if hb and hb.custom_mode == mode_id:
            print('mode ->', name, '(confirmed)')
            return
    raise MissionAbort(f'could not enter mode {name}')


def run():
    url = sys.argv[1] if len(sys.argv) > 1 else 'tcp:127.0.0.1:5762'
    m = mavutil.mavlink_connection(url, source_system=255)
    print(f'waiting for heartbeat on {url} ...')
    m.wait_heartbeat()
    print(f'connected (sys {m.target_system})')

    # SERIAL2 stream rates default to 0 -> raise them, plus interval requests
    for p, v in (('SR2_EXTRA1', 10), ('SR2_EXTRA2', 5), ('SR2_POSITION', 5)):
        m.mav.param_set_send(m.target_system, m.target_component,
                             p.encode(), float(v),
                             mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
        time.sleep(0.2)
    for msg_id, hz in ((30, 10), (137, 10), (33, 5)):  # ATTITUDE, SCALED_PRESSURE2, GPI
        m.mav.command_long_send(m.target_system, m.target_component,
                                mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
                                msg_id, int(1e6 / hz), 0, 0, 0, 0, 0)

    try:
        m.arducopter_arm()
        m.motors_armed_wait()
        print('ARMED')
        set_mode_confirmed(m, 'ALT_HOLD')

        manual_for(m, 2.0, label='settle')
        print(f'  telemetry types seen: {sorted(SEEN)}')
        if not SEEN & {'ATTITUDE', 'VFR_HUD', 'GLOBAL_POSITION_INT'}:
            raise MissionAbort('no telemetry streams on this link')

        print(f'commanding firmware depth target: {TARGET_DEPTH} m')
        t_end = time.time() + DIVE_TIMEOUT
        last = 0.0
        while time.time() < t_end:
            poll(m)
            set_target_depth(m, TARGET_DEPTH)          # resend ~10 Hz
            m.mav.manual_control_send(m.target_system, 0, 0, NEUTRAL_Z, 0, 0)
            d = STATE['depth']
            if time.time() - last > 0.5:
                last = time.time()
                status('dive', f'target={TARGET_DEPTH}')
            if d == d and abs(d - TARGET_DEPTH) < DEPTH_TOL:
                print(f'  reached {d:.2f} m')
                break
            time.sleep(0.1)
        else:
            print('  dive TIMEOUT - compare EKF depth above with bridge truth')

        manual_for(m, HOLD_TIME, label='hold')
        manual_for(m, LEG_TIME, x=FWD, label='forward')
        manual_for(m, LEG_TIME, y=FWD, label='right')
        manual_for(m, LEG_TIME, y=-FWD, label='left')
        manual_for(m, LEG_TIME, x=-FWD, label='backward')

        print('surfacing (firmware target 0.1 m)...')
        t_end = time.time() + DIVE_TIMEOUT
        while time.time() < t_end and not (STATE['depth'] < 0.2):
            poll(m)
            set_target_depth(m, 0.1)
            m.mav.manual_control_send(m.target_system, 0, 0, NEUTRAL_Z, 0, 0)
            time.sleep(0.1)
        manual_for(m, 1.0, label='neutral')
    except MissionAbort as e:
        print(f'\n!!! MISSION ABORTED: {e}')
        for _ in range(5):
            m.mav.manual_control_send(m.target_system, 0, 0, NEUTRAL_Z, 0, 0)
            time.sleep(0.1)
    finally:
        m.arducopter_disarm()
        m.motors_disarmed_wait()
        print('DISARMED - done')


def main():
    run()


if __name__ == '__main__':
    main()
