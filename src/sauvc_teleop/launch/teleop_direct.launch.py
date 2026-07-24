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
        # TELEOP HAS NO EVAL RUNNING. sim_drivers.launch.py's own default for
        # this argument is 'eval/ekf' — that topic only exists when
        # flow_eval_node is up. A pure teleop session (this launch file) never
        # runs it, so with the upstream default, depth_shim's x_est is DEAD on
        # arrival: same failure mode the FIX(dead x_est) comment there already
        # warns about for eval-less launches, just not covered by it. Default
        # here to ground truth instead, which sim_drivers always publishes
        # regardless of what else is running; override at the CLI if you
        # bring up an estimator alongside teleop and want the real x_est
        # behavior under test.
        DeclareLaunchArgument(
            'alt_odom_topic', default_value='/odometry',
            description='Odometry topic supplying x for the floor-profile '
                        'altitude lookup. Defaults to ground truth here '
                        '(teleop has no estimator running); pass '
                        'alt_odom_topic:=/eval/ekf or similar if one is up.'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(bridge_share, 'launch', 'sim_drivers.launch.py')),
            launch_arguments={
                'use_floor_profile': LaunchConfiguration('use_floor_profile'),
                'alt_odom_topic': LaunchConfiguration('alt_odom_topic'),
            }.items()),
        Node(package='sauvc_sim_bridge', executable='direct_control_node',
             name='direct_control_node', output='screen',
             parameters=[os.path.join(bridge_share, 'config', 'control_direct.yaml'),
                         {'cmd_z_is_depth': True}]),   # REQUIRED for absolute-mode teleop
    ])
