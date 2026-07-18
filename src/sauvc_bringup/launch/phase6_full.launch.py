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
            os.path.join(share, 'launch', 'phase5_ekf.launch.py'))),
        Node(package='sauvc_vision', executable='gate_detector_node'),
        Node(package='sauvc_mission', executable='mission_node'),
        # Logs the full run: fused pose + every prop detection, for post-run mission review.
        Node(package='sauvc_logging', executable='csv_logger_node',
             parameters=[{'topics': ['/odometry/filtered', '/vision/detections'],
                          'run_name': 'phase6_mission'}]),
    ])
