"""teleop_ardusub.launch.py — Path B stack for teleop: shims + ardusub_setpoint_node.

Does NOT start ArduSub SITL or ardusub_json_bridge.py -- start those first (see
sim_control_ardusub.launch.py's docstring). Does NOT launch the teleop node itself.

    (start SITL + json bridge per the sim README)
    Terminal 1: ros2 launch sauvc_teleop teleop_ardusub.launch.py
    Terminal 2: ros2 run sauvc_teleop keyboard_teleop_node --ros-args -p depth_mode:=pulse

In 'pulse' mode, r/f send a brief deflection and auto-return to neutral; ArduSub's own
ALT_HOLD does the actual holding. Arm and set ALT_HOLD yourself unless auto_arm:=true.
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
        DeclareLaunchArgument('auto_arm', default_value='false'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(bridge_share, 'launch', 'sim_drivers.launch.py')),
            launch_arguments={'use_floor_profile':
                              LaunchConfiguration('use_floor_profile')}.items()),
        Node(package='sauvc_sim_bridge', executable='ardusub_setpoint_node',
             name='ardusub_setpoint_node', output='screen',
             parameters=[os.path.join(bridge_share, 'config', 'control_ardusub.yaml'),
                         {'auto_arm': LaunchConfiguration('auto_arm')}]),
    ])
