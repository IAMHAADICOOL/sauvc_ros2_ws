#!/usr/bin/env python3
"""Pixhawk (ArduSub) IMU test via MAVLink: prints the FIRMWARE's view of the
vehicle - EKF attitude (roll/pitch/yaw), angular rates, and the raw IMU the
autopilot is consuming. This is what the real Pixhawk would report, as opposed
to the direct Stonefish sensor topic.

Prerequisites: sim + ArduSub SITL + JSON bridge running (see sauvc_stonefish
README, "ArduSub / Pixhawk in the loop"). pip install pymavlink.

Run:  ros2 run sauvc_sensor_tests pixhawk_imu_test
      (or plain: python3 pixhawk_imu_test.py [udp:127.0.0.1:14550])

MAVROS alternative: the Pixhawk IMU is also republished by MAVROS as a standard
sensor_msgs/Imu on /mavros/imu/data, so the existing node works unchanged:
      ros2 run sauvc_sensor_tests imu_test --ros-args -p topic:=/mavros/imu/data
"""
import math
import sys
from pymavlink import mavutil


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else 'udp:127.0.0.1:14550'
    m = mavutil.mavlink_connection(url)
    print(f'connecting to {url} - waiting for ArduSub heartbeat...')
    m.wait_heartbeat()
    print(f'connected (sys {m.target_system})')

    # Ask the autopilot to stream ATTITUDE (id 30) and RAW_IMU (id 27) at 20/10 Hz
    for msg_id, hz in ((30, 20), (27, 10)):
        m.mav.command_long_send(
            m.target_system, m.target_component,
            mavutil.mavlink.MAV_CMD_SET_MESSAGE_INTERVAL, 0,
            msg_id, int(1e6 / hz), 0, 0, 0, 0, 0)

    att, imu = None, None
    while True:
        msg = m.recv_match(type=('ATTITUDE', 'RAW_IMU'), blocking=True, timeout=5)
        if msg is None:
            print('\n(no data for 5 s - is the JSON bridge running?)')
            continue
        if msg.get_type() == 'ATTITUDE':
            att = msg
        else:
            imu = msg
        if att is None:
            continue
        line = (f'\rEKF RPY [deg]: {math.degrees(att.roll):+7.2f} '
                f'{math.degrees(att.pitch):+7.2f} {math.degrees(att.yaw):+7.2f} | '
                f'rates [rad/s]: {att.rollspeed:+6.3f} {att.pitchspeed:+6.3f} '
                f'{att.yawspeed:+6.3f}')
        if imu is not None:
            # RAW_IMU: accel in milli-g, gyro in mrad/s
            line += (f' | raw accel [m/s2]: {imu.xacc*9.81e-3:+7.3f} '
                     f'{imu.yacc*9.81e-3:+7.3f} {imu.zacc*9.81e-3:+7.3f}')
        print(line, end='', flush=True)


if __name__ == '__main__':
    main()
