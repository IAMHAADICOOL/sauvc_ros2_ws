from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    share = get_package_share_directory('sauvc_bringup')
    return LaunchDescription([
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(share, 'launch', 'phase3_flow.launch.py'))),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(share, 'launch', 'phase4_lane_heading.launch.py'))),
        Node(package='robot_localization', executable='ekf_node',
             name='ekf_filter_node',
             parameters=[os.path.join(share, 'config', 'ekf.yaml')]),
        # Auto-logs the fused pose/velocity for the square-test closure-error calc.
        Node(package='sauvc_logging', executable='csv_logger_node',
             parameters=[{'topics': ['/odometry/filtered'], 'run_name': 'phase5_ekf'}]),
    ])
