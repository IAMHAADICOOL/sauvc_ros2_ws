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
        Node(package='sauvc_localization', executable='preint_smoother_node'),
        # Logs BOTH estimators to the same run directory so the square-test / dropout-test
        # comparison in PIPELINE.md Phase 7 is a straight file diff, not two separate logs.
        Node(package='sauvc_logging', executable='csv_logger_node',
             parameters=[{'topics': ['/odometry/filtered', '/odometry/preint'],
                          'run_name': 'phase7_preint_ab'}]),
    ])
