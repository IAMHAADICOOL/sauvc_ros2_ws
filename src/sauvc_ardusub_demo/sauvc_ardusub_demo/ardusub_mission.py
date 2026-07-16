#!/usr/bin/env python3
"""Same mission as sauvc_motion_demo, but through ArduSub (the Pixhawk stack):
dive, depth-hold, forward/right/left/backward, surface - via MAVLink.

Here the depth "PID" is ArduSub's own ALT_HOLD controller, fed by the baro that
SITL synthesizes from the pressure/state our JSON bridge streams from Stonefish -
i.e. the firmware reads the (simulated) pressure sensor and closes the loop,
exactly like on the real vehicle.

Movement uses MANUAL_CONTROL, which per the ArduSub docs maps to the pilot input
channels: x = forward/back, y = lateral (right +), z = throttle/heave
(0..1000, 500 = neutral), r = yaw. (Equivalently you could publish RC overrides:
ch5 forward, ch6 lateral, ch3 throttle, ch4 yaw.)

Bring-up order:
  1. ros2 launch sauvc_stonefish sauvc_finals.launch.py
  2. sim_vehicle.py -v ArduSub -f json:127.0.0.1 --console      (FRAME_CONFIG = vectored-6DOF)
  3. ros2 run sauvc_stonefish ardusub_json_bridge.py
  4. ros2 run sauvc_ardusub_demo ardusub_mission
     (optional, in parallel: MAVROS -> ros2 run mavros mavros_node --ros-args -p fcu_url:=udp://:14550@
      then read the Pixhawk's own sensor view on /mavros/imu/data etc.)
"""
import time
from pymavlink import mavutil

NEUTRAL_Z = 500
DIVE_Z = 320          # z < 500 descends
SURFACE_Z = 700       # z > 500 ascends
FWD = 400             # -1000..1000
LEG_TIME = 5.0
DIVE_TIME = 4.0


def manual(m, x=0, y=0, z=NEUTRAL_Z, r=0, seconds=0.5):
    t_end = time.time() + seconds
    while time.time() < t_end:
        m.mav.manual_control_send(m.target_system, x, y, z, r, 0)
        time.sleep(0.1)


def set_mode(m, name):
    m.mav.set_mode_send(m.target_system,
                        mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                        m.mode_mapping()[name])
    print('mode ->', name)


def main():
    m = mavutil.mavlink_connection('udp:127.0.0.1:14550')
    print('waiting for ArduSub heartbeat...')
    m.wait_heartbeat()
    print(f'connected (sys {m.target_system})')

    set_mode(m, 'MANUAL')
    m.arducopter_arm()
    m.motors_armed_wait()
    print('ARMED')

    print('diving...')
    manual(m, z=DIVE_Z, seconds=DIVE_TIME)

    # Let the firmware hold depth (its baro = our simulated pressure sensor)
    set_mode(m, 'ALT_HOLD')
    print('depth hold for 5 s...')
    manual(m, seconds=5.0)

    print('forward...')
    manual(m, x=FWD, seconds=LEG_TIME)
    print('right (lateral)...')
    manual(m, y=FWD, seconds=LEG_TIME)
    print('left (lateral)...')
    manual(m, y=-FWD, seconds=LEG_TIME)
    print('backward...')
    manual(m, x=-FWD, seconds=LEG_TIME)

    print('surfacing...')
    set_mode(m, 'MANUAL')
    manual(m, z=SURFACE_Z, seconds=5.0)
    manual(m, seconds=1.0)

    m.arducopter_disarm()
    m.motors_disarmed_wait()
    print('DISARMED - mission complete')


if __name__ == '__main__':
    main()
