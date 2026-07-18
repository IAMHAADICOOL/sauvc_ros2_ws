from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(package='sauvc_localization', executable='lane_heading_node',
             parameters=[{'pool_axis_offset': 0.0, 'gain': 0.02}]),
        # Auto-logs both the raw line measurement and the corrected heading, so you can
        # compute detection rate (rows in line_meas vs pool_relative) and drift directly.
        Node(package='sauvc_logging', executable='csv_logger_node',
             parameters=[{'topics': ['/heading/pool_relative', '/heading/line_meas'],
                          'run_name': 'phase4_lane_heading'}]),
    ])
