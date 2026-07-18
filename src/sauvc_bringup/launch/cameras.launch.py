from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('sauvc_bringup'), 'config', 'cameras.yaml')
    return LaunchDescription([
        Node(package='v4l2_camera', executable='v4l2_camera_node',
             namespace='camera_down', name='camera_down', parameters=[cfg]),
        Node(package='v4l2_camera', executable='v4l2_camera_node',
             namespace='camera_front', name='camera_front', parameters=[cfg]),
    ])
