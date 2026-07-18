from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    """Pre-Phase convenience launch: brings up ALL sensor drivers + ALL check nodes at
    once. Use this only AFTER each sensor has been individually verified per SETUP.md's
    Pre-Phase section — debugging wiring problems one sensor at a time is much easier
    than debugging five at once. Override device paths via launch arguments; defaults
    match SETUP.md's examples.
    """
    share = get_package_share_directory('sauvc_bringup')
    return LaunchDescription([
        # Pressure sensor (source:=i2c default — Topology A, Bar30 direct to Jetson).
        # Topology B (Bar30 on the Pixhawk, read via pymavlink) CANNOT run in this
        # combined launch: it needs exclusive access to the Pixhawk link, which mavros
        # (below) is already using. Run it standalone, separately, if you're Topology B.
        Node(package='sauvc_sensor_check', executable='pressure_check',
             parameters=[{'source': 'i2c', 'i2c_bus': 1, 'sensor_model': 'bar30'}]),

        # Taobotics HFI-A9 IMU (remap publish_topic to /imu/data — see SETUP.md)
        # NOTE: /dev/imu_a9 assumes you've done section 4's udev pinning already. If not
        # yet pinned, override at the CLI: ... serial_port:=/dev/ttyUSB0
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            [os.path.join(get_package_share_directory('mrpt_sensor_imu_taobotics'),
                          'launch', 'mrpt_sensor_imu_taobotics.launch.py')]),
            launch_arguments={'serial_port': '/dev/imu_a9', 'sensor_model': 'hfi-a9',
                              'publish_topic': '/imu/data',
                              'sensor_frame_id': 'imu_link'}.items()),
        Node(package='sauvc_sensor_check', executable='imu_taobotics_check'),

        # Pixhawk IMU via mavros (assumes section 4's udev pinning; override fcu_url at
        # the CLI, e.g. fcu_url:=/dev/ttyACM0:57600, if not yet pinned)
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            [os.path.join(get_package_share_directory('mavros'),
                          'launch', 'apm.launch.py')]),
            launch_arguments={'fcu_url': '/dev/pixhawk:57600'}.items()),
        Node(package='sauvc_sensor_check', executable='imu_pixhawk_check'),

        # Cameras (v4l2 by default; see SETUP.md for the RealSense swap)
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(share, 'launch', 'cameras.launch.py'))),
        Node(package='sauvc_sensor_check', executable='camera_check_down'),
        Node(package='sauvc_sensor_check', executable='camera_check_front'),
    ])
