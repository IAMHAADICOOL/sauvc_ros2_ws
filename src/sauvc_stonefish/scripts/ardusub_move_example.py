#!/usr/bin/env python3
"""
Example: drive the simulated AUV through ArduSub (SITL) with pymavlink.

Prerequisites (3 terminals, in order):
  1. ros2 launch sauvc_stonefish sauvc_finals.launch.py
  2. sim_vehicle.py -v ArduSub -f json:127.0.0.1 --console
  3. ros2 run sauvc_stonefish ardusub_json_bridge.py
Then run this script:  python3 ardusub_move_example.py

It connects over MAVLink (same link MAVROS/QGC would use), arms in MANUAL mode,
and sends MANUAL_CONTROL commands - identical to what happens with the real
Pixhawk. Axes: x=forward/back, y=lateral, z=throttle/heave (0..1000, 500=neutral
for Sub), r=yaw. Range for x/y/r: -1000..1000.
"""
import time
from pymavlink import mavutil

m = mavutil.mavlink_connection('udp:127.0.0.1:14550')
print('waiting for heartbeat...')
m.wait_heartbeat()
print(f'connected: system {m.target_system}, component {m.target_component}')


def set_mode(mode_name):
    mode_id = m.mode_mapping()[mode_name]
    m.mav.set_mode_send(m.target_system,
                        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mode_id)
    print('mode ->', mode_name)


def manual(x=0, y=0, z=500, r=0, seconds=0.0):
    """Send MANUAL_CONTROL at 10 Hz for the given duration."""
    t_end = time.time() + seconds
    while time.time() < t_end:
        m.mav.manual_control_send(m.target_system, x, y, z, r, 0)
        time.sleep(0.1)


set_mode('MANUAL')          # try 'STABILIZE' or 'ALT_HOLD' once tuned
m.arducopter_arm()
m.motors_armed_wait()
print('ARMED')

print('diving for 4 s...')
manual(z=300, seconds=4)     # z<500 = descend

print('forward for 6 s...')
manual(x=500, seconds=6)

print('yawing right for 3 s...')
manual(r=400, seconds=3)

print('neutral for 2 s...')
manual(seconds=2)

m.arducopter_disarm()
m.motors_disarmed_wait()
print('DISARMED - done')
