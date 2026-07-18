"""sim_control_ardusub.launch.py — PATH B: control THROUGH ArduSub SITL.

    mission_node --/cmd/setpoint--> ardusub_setpoint_node --MAVLink--> ArduSub SITL
    ArduSub SITL <--JSON--> ardusub_json_bridge <--> Stonefish thrusters + state

This launch starts the shim drivers + the MAVLink setpoint bridge ONLY. It does NOT start
ArduSub SITL or ardusub_json_bridge.py — those are separate processes with their own
setup (see the sauvc_stonefish README "ArduSub / Pixhawk in the loop"). Start them first:

    1. ArduSub SITL   (sim_vehicle.py -v ArduSub -f json ...)
    2. ros2 run sauvc_stonefish ardusub_json_bridge        (or however it's launched)
    3. ros2 launch sauvc_sim_bridge sim_control_ardusub.launch.py

The shim drivers here publish /imu/data, /depth, /altitude for the LOCALIZATION stack.
Note the autopilot runs its OWN EKF off the JSON-backend IMU — that is independent of your
robot_localization EKF, and the two are allowed to disagree (see pixhawk_imu_test notes).
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
        DeclareLaunchArgument('auto_arm', default_value='false'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(share, 'launch', 'sim_drivers.launch.py')),
            launch_arguments={'use_floor_profile':
                              LaunchConfiguration('use_floor_profile')}.items()),
        Node(package='sauvc_sim_bridge', executable='ardusub_setpoint_node',
             name='ardusub_setpoint_node', output='screen',
             parameters=[os.path.join(share, 'config', 'control_ardusub.yaml'),
                         {'auto_arm': LaunchConfiguration('auto_arm')}]),
    ])
