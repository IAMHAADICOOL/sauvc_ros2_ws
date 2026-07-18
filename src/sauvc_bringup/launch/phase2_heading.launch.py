from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    # Assumes an IMU driver already publishes /imu/data_raw (see SETUP.md for your IMU).
    # If your IMU outputs a fused quaternion on /imu/data natively, skip this filter.
    return LaunchDescription([
        Node(package='imu_filter_madgwick', executable='imu_filter_madgwick_node',
             parameters=[{'use_mag': False, 'publish_tf': False,
                          'world_frame': 'enu'}],
             remappings=[('imu/data_raw', '/imu/data_raw'),
                         ('imu/data', '/imu/data')]),
        # Auto-logs /imu/data; Ctrl-C prints mean/std per column — use wz's std for the
        # Phase 2b gyro variance, and eyeball qz drift over the 10-min stationary hold.
        Node(package='sauvc_logging', executable='csv_logger_node',
             parameters=[{'topics': ['/imu/data'], 'run_name': 'phase2_heading'}]),
    ])
