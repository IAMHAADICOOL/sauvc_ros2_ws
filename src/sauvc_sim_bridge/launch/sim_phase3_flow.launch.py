"""sim_phase3_flow.launch.py — Phase 3 (optical-flow velocity) in simulation.

Sim twin of sauvc_bringup/launch/phase3_flow.launch.py. flow_velocity_node is the SAME
executable from the SAME package; only flow_sim.yaml differs.

Unlike hardware, you get a scored answer: flow_scorer_node grades /flow/twist against
the simulated DVL and prints the scale error your Phase 3 spec asks for. The DVL never
leaves the /sauvc_auv namespace and is never fused.
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
    robot = LaunchConfiguration('robot_name')

    return LaunchDescription([
        DeclareLaunchArgument('robot_name', default_value='sauvc_auv'),
        DeclareLaunchArgument('score', default_value='true'),

        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(share, 'launch', 'sim_drivers.launch.py'))),

        Node(package='sauvc_localization', executable='flow_velocity_node',
             name='flow_velocity_node', output='screen',
             parameters=[os.path.join(share, 'config', 'flow_sim.yaml')],
             # No remap needed: image_relay_node (in sim_drivers) already publishes
             # /camera_down/image_raw from the sim's image_color. flow_sim.yaml's
             # image_topic points at that relayed name.
             ),

        Node(package='sauvc_sim_bridge', executable='flow_scorer_node',
             name='flow_scorer_node', output='screen',
             parameters=[{'dvl_topic': ['/', robot, '/dvl']}]),

        Node(package='sauvc_logging', executable='csv_logger_node',
             name='csv_logger_node',
             parameters=[{'topics': ['/flow/twist', '/altitude', '/sim/rtf'],
                          'run_name': 'sim_phase3_flow'}]),
    ])
