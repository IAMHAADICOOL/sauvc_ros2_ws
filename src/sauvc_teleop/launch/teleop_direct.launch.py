"""teleop_direct.launch.py — Path A stack for teleop: shims + direct_control_node.

Brings up sim_drivers + direct_control_node with cmd_z_is_depth FORCED true, because
'absolute' depth-hold teleop only makes sense against an absolute-depth setpoint
controller. Does NOT launch the teleop node itself -- run that separately (see
keyboard_teleop_node's docstring for why).

    Terminal 1: ros2 launch sauvc_teleop teleop_direct.launch.py
    Terminal 2: ros2 run sauvc_teleop keyboard_teleop_node --ros-args -p depth_mode:=absolute
"""
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    bridge_share = get_package_share_directory('sauvc_sim_bridge')
    return LaunchDescription([
        DeclareLaunchArgument('use_floor_profile', default_value='true'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(bridge_share, 'launch', 'sim_drivers.launch.py')),
            launch_arguments={'use_floor_profile':
                              LaunchConfiguration('use_floor_profile')}.items()),
        Node(package='sauvc_sim_bridge', executable='direct_control_node',
             name='direct_control_node', output='screen',
             parameters=[os.path.join(bridge_share, 'config', 'control_direct.yaml'),
                         {'cmd_z_is_depth': True}]),   # REQUIRED for absolute-mode teleop
    ])
