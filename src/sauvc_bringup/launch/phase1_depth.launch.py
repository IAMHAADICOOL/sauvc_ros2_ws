from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='sauvc_drivers', executable='depth_altitude_node',
             # use_floor_profile toggles the altitude source:
             #   False -> FLAT test mode: altitude = pool_depth - depth. Use this in your
             #            constant-depth practice pool, with pool_depth set to ITS depth.
             #   True  -> SAUVC mode: V-profile from the rulebook side view (1.2 m at the
             #            walls, 1.6 m mid-length of 25 m), evaluated at the EKF's x.
             #            Confirm numbers at the venue (+/-5% rulebook tolerance).
             # Override per-run without editing this file:
             #   ros2 launch sauvc_bringup phase1_depth.launch.py  (defaults below)
             #   ... or ros2 run sauvc_drivers depth_altitude_node --ros-args \
             #          -p use_floor_profile:=false -p pool_depth:=2.0
             # sensor_model MUST match your actual board ('bar02' or 'bar30').
             # depth_var is a PLACEHOLDER — overwrite with your Phase 1 measured std^2.
             parameters=[{'use_floor_profile': False,   # <-- flip to True for SAUVC
                          'pool_depth': 1.4,            # <-- your practice pool's depth
                          'floor_profile_x': [0.0, 12.5, 25.0],
                          'floor_profile_depth': [1.2, 1.6, 1.2],
                          'i2c_bus': 1, 'sensor_model': 'bar30',
                          'rate_hz': 20.0, 'depth_var': 0.0004}]),
        # Auto-logs every /depth and /altitude sample to CSV; Ctrl-C prints mean/std —
        # exactly the numbers Phase 1's surface/0.5m/1.0m holds ask for.
        Node(package='sauvc_logging', executable='csv_logger_node',
             parameters=[{'topics': ['/depth', '/altitude'], 'run_name': 'phase1_depth'}]),
    ])
