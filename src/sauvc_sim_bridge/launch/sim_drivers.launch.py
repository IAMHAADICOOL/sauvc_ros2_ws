"""sim_drivers.launch.py — the sim's replacement for sauvc_drivers + cameras.launch.py.

This is the ONLY seam between the simulator and your stack. Include it instead of
phase1_depth.launch.py + cameras.launch.py, and every downstream node runs unmodified.

After this launch, the following topics exist and are indistinguishable from hardware:
    /imu/data                 (ENU/FLU, from the NED/FRD sim IMU)
    /depth, /altitude         (from the sim pressure sensor, surface-zeroed)
    /camera_down/image_raw    (relayed from /sauvc_auv/camera_down/image_color)
    /camera_front/image_raw   (relayed from /sauvc_auv/camera_front/image_color)

use_sim_time is deliberately NOT set. Stonefish publishes no /clock and stamps every
message with the wall clock (ROS2Interface.cpp: get_clock()->now()). Setting
use_sim_time:=true here would starve every node of time and hang the stack.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot = LaunchConfiguration('robot_name')
    use_profile = LaunchConfiguration('use_floor_profile')
    alt_odom = LaunchConfiguration('alt_odom_topic')

    return LaunchDescription([
        DeclareLaunchArgument('robot_name', default_value='sauvc_auv'),
        DeclareLaunchArgument(
            'use_floor_profile', default_value='true',
            description='true = SAUVC V-floor; false = flat pool. Must match the scene '
                        'you launched: sauvc_pool.scn is flat, the competition arenas '
                        'are V-shaped.'),
        # FIX(dead x_est): where depth_shim gets the x that drives the floor-profile
        # altitude. Default keeps the hardware-parity topic; eval launches that run
        # WITHOUT robot_localization must override this (ground truth is fine for a
        # diagnostic), otherwise x_est never updates and /altitude is wrong.
        # DeclareLaunchArgument(
        #     'alt_odom_topic', default_value='/odometry/filtered',
        #     description='Odometry topic supplying x for the floor-profile lookup.'),
        
        DeclareLaunchArgument(
                    'alt_odom_topic', default_value='/odometry/filtered',
                    description='Odometry topic supplying x for the floor-profile lookup.'),

        # Republish /sauvc_auv/*/image_color -> /camera_*/image_raw. lane_heading_node
        # and gate_detector_node HARDCODE the image_raw names (no param), so a relay is
        # the only thing that satisfies all three camera consumers without editing them.
        Node(package='sauvc_sim_bridge', executable='image_relay_node',
             name='image_relay_node', output='screen',
             parameters=[{'down_src':  ['/', robot, '/camera_down/image_color'],
                          'front_src': ['/', robot, '/camera_front/image_color']}]),

        Node(package='sauvc_sim_bridge', executable='imu_shim_node',
             name='imu_shim_node', output='screen',
             parameters=[{'in_topic': ['/', robot, '/imu'], 'out_topic': '/imu/data'}]),

        Node(package='sauvc_sim_bridge', executable='depth_shim_node',
             name='depth_shim_node', output='screen',
             parameters=[{'in_topic': ['/', robot, '/pressure'],
                          'use_floor_profile': use_profile,
                          'pool_depth': 1.4,
                          # MUST match <water density="1000.0"/> in the scene and
                          # Stonefish's g -- NOT the real pool's 997.0 / 9.80665.
                          # Using the real robot's constants here injects a silent
                          # +0.335% depth scale error. Plant parameters belong to
                          # the plant, and this plant is the simulator.
                          'fluid_density': 1000.0,
                          'gravity': 9.81,
                          'floor_profile_x': [0.0, 12.5, 25.0],
                          'floor_profile_depth': [1.2, 1.6, 1.2],
                          # world x (-12.5..+12.5) -> wall-referenced profile x (0..25)
                          'profile_x_offset': 12.5,
                          # mount offsets from my_auv.scn (FRD): pressure sensor
                          # z=-0.10 (above origin), down camera z=+0.11 (below).
                          # MEASURE these on the real vehicle; parity depends on it.
                          'sensor_above_origin': 0.10,
                          'camera_below_origin': 0.11,
                          'odom_topic': alt_odom}]),

        # RTF monitor is NOT optional. See rtf_monitor_node's docstring: below RTF 1.0
        # the optical-flow velocity is silently scaled and the EKF fuses inconsistent
        # kinematics. declared_odom_rate must match rate="..." on the odometry sensor.
        Node(package='sauvc_sim_bridge', executable='rtf_monitor_node',
             name='rtf_monitor_node', output='screen',
             parameters=[{'odom_topic': ['/', robot, '/odometry'],
                          'declared_odom_rate': 30.0}]),
    ])
