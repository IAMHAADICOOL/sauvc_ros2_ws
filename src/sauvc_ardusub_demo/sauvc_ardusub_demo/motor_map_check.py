#!/usr/bin/env python3
"""Automated MOTOR_MAP / MOTOR_SIGN checker.

For each ArduSub motor 1..8 it commands the firmware's own MOTOR_TEST
(slightly positive, then slightly negative) and watches the Stonefish
/<robot>/thruster_state topic to see WHICH simulated thruster actually spun
and with what sign. At the end it prints the exact MOTOR_MAP and MOTOR_SIGN
lines to paste into ardusub_json_bridge.py.

Run order (4 terminals):
  1. ros2 launch sauvc_stonefish sauvc_finals.launch.py
  2. sim_vehicle.py -v ArduSub -f json:127.0.0.1 --console
  3. ros2 run sauvc_stonefish ardusub_json_bridge.py
  4. ros2 run sauvc_ardusub_demo motor_map_check

Notes:
- The vehicle will twitch during the test; that's fine. Best done while it
  floats free (not pressed against the frame).
- If thruster_state isn't available in your stonefish_ros2 build, the tool
  falls back to odometry (less direct: reports body accelerations instead).
"""
import threading
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from pymavlink import mavutil

STONEFISH_ORDER = ['HFP', 'HFS', 'HAP', 'HAS', 'VFP', 'VFS', 'VAP', 'VAS']
TEST_THROTTLE_POS = 60.0   # percent; 50 = neutral for Sub
TEST_THROTTLE_NEG = 40.0
TEST_SECONDS = 2.0
SETTLE_SECONDS = 1.5


class ThrusterWatch(Node):
    def __init__(self):
        super().__init__('motor_map_check')
        self.declare_parameter('robot', 'sauvc_auv')
        robot = self.get_parameter('robot').value
        self.rpms = None
        self.have_state = False
        try:
            from stonefish_ros2.msg import ThrusterState
            self.create_subscription(ThrusterState, f'/{robot}/thruster_state',
                                     self.ts_cb, qos_profile_sensor_data)
            self.msg_ok = True
        except ImportError:
            self.get_logger().warn('stonefish_ros2 msgs not importable - '
                                   'falling back to setpoint echo')
            self.msg_ok = False
            from std_msgs.msg import Float64MultiArray
            self.create_subscription(Float64MultiArray,
                                     f'/{robot}/thruster_setpoints',
                                     self.sp_cb, qos_profile_sensor_data)

    def ts_cb(self, msg):
        # Prefer thrust[] - its sign IS the thrust direction. rpm sign is the
        # SPIN direction, which is inverted on the inverted_setpoint (LH)
        # thrusters and misled the first analysis.
        if hasattr(msg, 'thrust') and len(msg.thrust):
            self.rpms = list(msg.thrust)
        elif hasattr(msg, 'rpm') and len(msg.rpm):
            self.rpms = list(msg.rpm)
        else:
            self.rpms = list(msg.setpoint)
        self.have_state = True

    def sp_cb(self, msg):
        self.rpms = list(msg.data)
        self.have_state = True

    def snapshot(self):
        return list(self.rpms) if self.rpms else None


def motor_test(m, motor_1based, throttle_pct, seconds):
    """Send DO_MOTOR_TEST and return the firmware's ACK result string.
    NOTE: ArduPilot only allows motor test while DISARMED (it drives the
    motors itself). An armed vehicle silently produced the all-zero run."""
    m.mav.command_long_send(
        m.target_system, m.target_component,
        mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST, 0,
        motor_1based,                                   # param1: motor instance
        mavutil.mavlink.MOTOR_TEST_THROTTLE_PERCENT,    # param2: test type
        throttle_pct,                                   # param3: throttle
        seconds,                                        # param4: timeout
        0, 0, 0)
    # Only accept the ACK for OUR command; MAVProxy's own background traffic
    # (message-interval requests etc.) also produces COMMAND_ACKs.
    deadline = time.time() + 3.0
    while time.time() < deadline:
        ack = m.recv_match(type='COMMAND_ACK', blocking=True, timeout=0.5)
        if ack is None:
            continue
        if ack.command != mavutil.mavlink.MAV_CMD_DO_MOTOR_TEST:
            continue
        results = {0: 'ACCEPTED', 1: 'TEMPORARILY_REJECTED', 2: 'DENIED',
                   3: 'UNSUPPORTED', 4: 'FAILED', 5: 'IN_PROGRESS'}
        return results.get(ack.result, f'RESULT_{ack.result}')
    return 'NO_ACK'


def set_param(m, name, value, timeout=3):
    m.mav.param_set_send(m.target_system, m.target_component,
                         name.encode(), float(value),
                         mavutil.mavlink.MAV_PARAM_TYPE_REAL32)
    deadline = time.time() + timeout
    while time.time() < deadline:
        msg = m.recv_match(type='PARAM_VALUE', blocking=True, timeout=0.5)
        if msg:
            pid = msg.param_id
            if isinstance(pid, bytes):
                pid = pid.decode(errors='ignore')
            if pid.strip('\x00') == name:
                return msg.param_value
    return None


def get_param(m, name, timeout=3):
    m.mav.param_request_read_send(m.target_system, m.target_component,
                                  name.encode(), -1)
    msg = m.recv_match(type='PARAM_VALUE', blocking=True, timeout=timeout)
    return None if msg is None else msg.param_value


def dominant(delta):
    idx = max(range(len(delta)), key=lambda i: abs(delta[i]))
    return idx, delta[idx]


def group_fallback(m, watch):
    """Arm in MANUAL and command one pilot axis at a time via MANUAL_CONTROL,
    reporting the per-thruster response. Doesn't isolate single motors, but
    tells you whether the current MOTOR_MAP produces sane group behavior:
      forward  -> only HFP,HFS,HAP,HAS (indices 0-3), all same sign
      lateral  -> only 0-3, split by side
      throttle -> only VFP,VFS,VAP,VAS (indices 4-7), all same sign
      yaw      -> only 0-3, split diagonally"""
    m.mav.set_mode_send(m.target_system,
                        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                        m.mode_mapping()['MANUAL'])
    m.arducopter_arm()
    m.motors_armed_wait()
    print('ARMED (MANUAL) for group test')
    axes = [('forward+', dict(x=400)), ('lateral+ (right)', dict(y=400)),
            ('throttle+ (up)', dict(z=700)), ('throttle- (down)', dict(z=300)),
            ('yaw+ (right)', dict(r=400))]
    for name, kw in axes:
        time.sleep(SETTLE_SECONDS)
        base = watch.snapshot()
        t_end = time.time() + 2.0
        while time.time() < t_end:
            m.mav.manual_control_send(m.target_system, kw.get('x', 0),
                                      kw.get('y', 0), kw.get('z', 500),
                                      kw.get('r', 0), 0)
            time.sleep(0.1)
        during = watch.snapshot()
        # neutral stick to settle
        for _ in range(5):
            m.mav.manual_control_send(m.target_system, 0, 0, 500, 0, 0)
            time.sleep(0.1)
        delta = [round(d - b, 1) for d, b in zip(during, base)]
        active = {STONEFISH_ORDER[i]: v for i, v in enumerate(delta)
                  if abs(v) > 0.5}
        print(f'  {name:18s} -> {active if active else "no response"}')
    m.arducopter_disarm()
    print('Compare against the expected groups in this function docstring; '
          'wrong members/signs identify the MOTOR_MAP/MOTOR_SIGN entries to fix.')


def main():
    rclpy.init()
    watch = ThrusterWatch()
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(watch)
    spin = threading.Thread(target=executor.spin, daemon=True)
    spin.start()
    try:
        _run(watch)
    finally:
        executor.shutdown()
        watch.destroy_node()
        rclpy.shutdown()
        spin.join(timeout=2)


def _run(watch):

    m = mavutil.mavlink_connection('udp:127.0.0.1:14550')
    print('waiting for ArduSub heartbeat...')
    m.wait_heartbeat()
    print(f'connected (sys {m.target_system}); waiting for thruster feedback...')
    while not watch.have_state:
        time.sleep(0.2)

    # Frame sanity: vectored-6DOF (8 motors) is FRAME_CONFIG=2 on ArduSub.
    fc = get_param(m, 'FRAME_CONFIG')
    print(f'FRAME_CONFIG = {fc}')
    if fc is not None and int(fc) != 2:
        print('*** FRAME_CONFIG is NOT 2 (Vectored-6DOF, 8 motors). Your bridge')
        print('*** log showing pwm=[1500 x6, 0, 0] means only 6 motors exist.')
        print('*** In the SITL console run:   param set FRAME_CONFIG 2')
        print('*** then RESTART SITL (the motor mixer is built at boot), and')
        print('*** rerun this tool. Continuing anyway for reference...')

    # ArduSub's motor test ARMS INTERNALLY and therefore runs the arming
    # checks; in SITL with no RC/joystick configured they fail -> result
    # FAILED. Disable checks for this mapping session and restore after.
    old_check = get_param(m, 'ARMING_CHECK')
    print(f'ARMING_CHECK = {old_check} -> setting 0 for the mapping session')
    if set_param(m, 'ARMING_CHECK', 0) is None:
        print('  (could not set ARMING_CHECK - is the sim+bridge feeding SITL?)')

    # DO NOT arm ourselves: ArduPilot rejects MOTOR_TEST while already armed.
    print('starting per-motor tests (vehicle stays DISARMED)\n')

    mapping, signs = [], []
    for motor in range(1, 9):
        time.sleep(SETTLE_SECONDS)
        base = watch.snapshot()
        ack = motor_test(m, motor, TEST_THROTTLE_POS, TEST_SECONDS)
        if ack != 'ACCEPTED':
            print(f'ArduSub motor {motor}: MOTOR_TEST {ack} - no data '
                  f'(vehicle must be disarmed; check FRAME_CONFIG)')
            mapping.append(-1)
            signs.append(0)
            continue
        time.sleep(TEST_SECONDS * 0.7)
        during = watch.snapshot()
        time.sleep(TEST_SECONDS * 0.5 + SETTLE_SECONDS)

        delta = [d - b for d, b in zip(during, base)]
        idx, val = dominant(delta)
        others = sorted((abs(v) for i, v in enumerate(delta) if i != idx),
                        reverse=True)
        clean = not others or others[0] < 0.3 * abs(val) if val else False
        if abs(val) < 1e-6:
            print(f'ArduSub motor {motor}: ACK ACCEPTED but NO thruster responded '
                  f'- this motor channel is unused by the current frame '
                  f'(FRAME_CONFIG wrong) or the bridge is not running')
            mapping.append(-1)
            signs.append(0)
            continue
        mapping.append(idx)
        signs.append(1 if val > 0 else -1)
        warn = '' if clean else '  (WARNING: not clean - other thrusters also moved)'
        print(f'ArduSub motor {motor}: -> Stonefish [{idx}] {STONEFISH_ORDER[idx]} '
              f'response {val:+.1f}{warn}')
        print(f'    raw delta: {[round(v, 1) for v in delta]}')

    if old_check is not None:
        set_param(m, 'ARMING_CHECK', old_check)
        print(f'\nARMING_CHECK restored to {old_check}')

    if all(x < 0 for x in mapping):
        print('\nMotor test never ran - falling back to ARMED GROUP TEST:')
        print('commanding pure pilot axes and reporting which thrusters respond.')
        group_fallback(m, watch)

    print('\n================ RESULT ================')
    if any(x < 0 for x in mapping):
        print('INCOMPLETE RUN - some motors gave no data (see messages above).')
        print('Most common fix: param set FRAME_CONFIG 2, restart SITL, rerun.')
        print(f'partial MOTOR_MAP  = {mapping}   (-1 = no data)')
        print(f'partial MOTOR_SIGN = {signs}')
    else:
        print('Paste into sauvc_stonefish/scripts/ardusub_json_bridge.py:')
        print(f'MOTOR_MAP  = {mapping}')
        print(f'MOTOR_SIGN = {signs}')
        if len(set(mapping)) != 8:
            print('WARNING: duplicate assignments - two motors drove the same '
                  'thruster. Check FRAME_CONFIG and rerun.')
    print('NOTE on SIGN: +1 means positive PWM produced positive thrust on that '
          'thruster. ArduSub expects specific directions per motor; if the '
          'vehicle still fights itself in STABILIZE/ALT_HOLD after mapping, '
          'flip the offending MOTOR_SIGN entries (or MOT_n_DIRECTION params).')


if __name__ == '__main__':
    main()
