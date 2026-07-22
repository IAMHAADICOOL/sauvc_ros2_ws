"""record_trajectory_sim.launch.py — one trajectory per launch, SIMULATION side.

    ros2 launch sauvc_traj_recorder record_trajectory_sim.launch.py

Records straight off Stonefish's own topics (/<robot>/camera_front/image_color, etc.),
so it works standalone against any of the sauvc_stonefish scenarios — no sim_drivers,
no shims needed. Includes /<robot>/odometry (ground truth), which only exists in sim;
see record_trajectory_hw.launch.py for the real vehicle, which has no such topic.

Bring up the scenario (e.g. sauvc_qualification.launch.py) and your teleop separately,
then run this once per take. Ctrl+C to stop, relaunch for the next one — the trajectory
number auto-increments by scanning base_dir.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

from sauvc_traj_recorder.recording_common import resolve_trajectory_dir, build_recording_actions


def setup(context, *a, **k):
    robot = LaunchConfiguration('robot_name').perform(context)
    base_dir = LaunchConfiguration('base_dir').perform(context)
    forced_id = LaunchConfiguration('trajectory_id').perform(context)
    image_format = LaunchConfiguration('image_format').perform(context)
    video_fps = LaunchConfiguration('video_fps').perform(context)
    record_bag = LaunchConfiguration('record_bag').perform(context).lower() == 'true'

    traj_dir, traj_name, already_existed = resolve_trajectory_dir(base_dir, forced_id)

    topics = {
        'front_image_topic': f'/{robot}/camera_front/image_color',
        'front_info_topic': f'/{robot}/camera_front/camera_info',
        'down_image_topic': f'/{robot}/camera_down/image_color',
        'down_info_topic': f'/{robot}/camera_down/camera_info',
        'imu_topic': f'/{robot}/imu',
        'pressure_topic': f'/{robot}/pressure',
        'odom_topic': f'/{robot}/odometry',
    }
    node_params = dict(topics)
    node_params.update({
        'output_dir': traj_dir,
        'image_format': image_format,
        'video_fps': float(video_fps),
    })

    return build_recording_actions(
        traj_dir, traj_name, already_existed, list(topics.values()),
        node_params, record_bag, 'sim')


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'robot_name', default_value='sauvc_auv',
            description='must match robot_name in the launched .scn'),
        DeclareLaunchArgument(
            'base_dir', default_value='/home/haadi/Robotics_Job/sauvc_ws/src/sauvc_traj_recorder/sauvc_data/trajectories',
            description='parent folder; trajectory_NN/ subfolders are created inside it'),
        DeclareLaunchArgument(
            'trajectory_id', default_value='',
            description='force a specific NN instead of auto-incrementing '
                        '(e.g. 5 -> trajectory_05); leave empty to auto-increment'),
        DeclareLaunchArgument(
            'image_format', default_value='png',
            description='png (lossless) or jpg (smaller, lossy)'),
        DeclareLaunchArgument(
            'video_fps', default_value='10.0',
            description='nominal fps for the preview .mp4 (container needs a fixed '
                        'value; real capture rate is irregular — see *_index.csv)'),
        DeclareLaunchArgument(
            'record_bag', default_value='true',
            description='also record a replayable rosbag2 bag alongside image/CSV logs'),
        OpaqueFunction(function=setup),
    ])
