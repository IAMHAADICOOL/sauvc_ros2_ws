"""sim_control_direct.launch.py — PATH A: direct PID control, no ArduSub.

    mission_node --/cmd/setpoint--> direct_control_node --> /sauvc_auv/thruster_setpoints

Brings up the shim drivers + the direct controller. Add mission_node (or drive
/cmd/setpoint by hand with `ros2 topic pub`) on top. This is the fast, deterministic path
for developing localization + mission logic.
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    share = get_package_share_directory('sauvc_sim_bridge')
    return LaunchDescription([
        DeclareLaunchArgument('use_floor_profile', default_value='true'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(share, 'launch', 'sim_drivers.launch.py')),
            launch_arguments={'use_floor_profile':
                              LaunchConfiguration('use_floor_profile')}.items()),
        Node(package='sauvc_sim_bridge', executable='direct_control_node',
             name='direct_control_node', output='screen',
             parameters=[os.path.join(share, 'config', 'control_direct.yaml')]),
    ])
