"""flow_eval.launch.py — the comparison node + the shim drivers it needs.

Brings up sim_drivers (imu_shim, depth_shim, image_relay, rtf_monitor) and the eval node.
Does NOT start Stonefish, control, or teleop — run those separately (see README).
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    bridge = get_package_share_directory('sauvc_sim_bridge')
    return LaunchDescription([
        DeclareLaunchArgument('compare_frame', default_value='ned'),
        DeclareLaunchArgument('use_floor_profile', default_value='true'),
        DeclareLaunchArgument('show_windows', default_value='true'),
        DeclareLaunchArgument('show_optical_flow', default_value='true'),
        DeclareLaunchArgument('show_camera', default_value='true'),
        DeclareLaunchArgument('print_estimates', default_value='true'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(bridge, 'launch', 'sim_drivers.launch.py')),
            launch_arguments={'use_floor_profile':
                              LaunchConfiguration('use_floor_profile')}.items()),
        Node(package='sauvc_flow_eval', executable='flow_eval_node',
             name='flow_eval_node', output='screen',
             parameters=[{'compare_frame': LaunchConfiguration('compare_frame'),
                          'show_windows': LaunchConfiguration('show_windows'),
                          'show_optical_flow': LaunchConfiguration('show_optical_flow'),
                          'show_camera': LaunchConfiguration('show_camera'),
                          'print_estimates': LaunchConfiguration('print_estimates')}]),
    ])
