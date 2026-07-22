"""record_trajectory_hw.launch.py — one trajectory per launch, REAL ROBOT side.

    ros2 launch sauvc_traj_recorder record_trajectory_hw.launch.py

Records the topics the real driver stack actually publishes: /camera_front/image_raw,
/camera_down/image_raw, /imu/data, /depth, /altitude. No odom_topic here — ground truth
odometry is a sim-only reference (see sauvc_sim_bridge's DVL comments); the real vehicle
has nothing to fill that role, so this launch simply doesn't record it. If your setup has
an external ground-truth source (mocap, USBL fix log, etc.), record it separately and
join on timestamp afterwards — this package doesn't assume one exists.

There's no raw /pressure topic on hardware by design (the Bar30 is read directly inside
depth_altitude_node over I2C, which only ever publishes the derived /depth + /altitude —
see depth_shim_node's docstring in sauvc_sim_bridge for the sim-side mirror of this).

ASSUMES: your camera driver publishes CameraInfo alongside each image_raw topic under the
standard <ns>/camera_info convention. If yours doesn't, override front_info_topic /
down_info_topic to '' via the recorder node's parameters (or edit this file) and you'll
just get images without an intrinsics snapshot — everything else is unaffected.

Does NOT bring up the camera/IMU/depth drivers or teleop — start those first (whatever
your normal hardware bringup is), then run this once per take. Ctrl+C to stop, relaunch
for the next one — the trajectory number auto-increments by scanning base_dir.
"""
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

from sauvc_traj_recorder.recording_common import resolve_trajectory_dir, build_recording_actions


def setup(context, *a, **k):
    base_dir = LaunchConfiguration('base_dir').perform(context)
    forced_id = LaunchConfiguration('trajectory_id').perform(context)
    image_format = LaunchConfiguration('image_format').perform(context)
    video_fps = LaunchConfiguration('video_fps').perform(context)
    record_bag = LaunchConfiguration('record_bag').perform(context).lower() == 'true'

    traj_dir, traj_name, already_existed = resolve_trajectory_dir(base_dir, forced_id)

    topics = {
        'front_image_topic': '/camera_front/image_raw',
        'front_info_topic': '/camera_front/camera_info',
        'down_image_topic': '/camera_down/image_raw',
        'down_info_topic': '/camera_down/camera_info',
        'imu_topic': '/imu/data',
        'depth_topic': '/depth',
        'altitude_topic': '/altitude',
        # no odom_topic — no ground truth on hardware
    }
    node_params = dict(topics)
    node_params.update({
        'output_dir': traj_dir,
        'image_format': image_format,
        'video_fps': float(video_fps),
    })

    return build_recording_actions(
        traj_dir, traj_name, already_existed, list(topics.values()),
        node_params, record_bag, 'hardware')


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument(
            'base_dir', default_value='~/sauvc_data/trajectories',
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
