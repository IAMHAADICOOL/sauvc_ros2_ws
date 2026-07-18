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
            os.path.join(share, 'launch', 'cameras.launch.py'))),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(share, 'launch', 'phase1_depth.launch.py'))),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(share, 'launch', 'phase2_heading.launch.py'))),
        Node(package='sauvc_localization', executable='flow_velocity_node',
             parameters=[os.path.join(share, 'config', 'flow.yaml')]),
        # Auto-logs /flow/twist; Ctrl-C prints vx/vy mean+std directly — that's the
        # stationary-noise number and (via the mean during a push test) the bias check.
        Node(package='sauvc_logging', executable='csv_logger_node',
             parameters=[{'topics': ['/flow/twist'], 'run_name': 'phase3_flow'}]),
    ])
