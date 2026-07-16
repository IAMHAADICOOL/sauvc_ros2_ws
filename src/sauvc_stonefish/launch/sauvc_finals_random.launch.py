"""Launch the finals arena with RANDOMIZED prop positions.

    ros2 launch sauvc_stonefish sauvc_finals_random.launch.py seed:=42

Same seed -> identical arena every launch (reproducible runs); change the seed
for a new competition draw. The randomized scene is generated in /tmp at launch
time; the installed sauvc_finals.scn is never modified.
"""
import os
import sys

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def setup(context, *a, **k):
    seed = int(LaunchConfiguration('seed').perform(context))
    share = get_package_share_directory('sauvc_stonefish')
    src = os.path.join(share, 'scenarios', 'sauvc_finals.scn')
    out = f'/tmp/sauvc_finals_seed{seed}.scn'

    sys.path.insert(0, os.path.join(share, 'scripts'))
    from randomize_arena import randomize
    text, layout = randomize(open(src).read(), seed)
    open(out, 'w').write(text)
    print(f'[randomize_arena] {layout}')

    return [IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('stonefish_ros2'),
                                  'launch', 'stonefish_simulator.launch.py'])
        ]),
        launch_arguments={
            'simulation_data': os.path.join(share, 'data'),
            'scenario_desc': out,
            'simulation_rate': '300.0',
            'window_res_x': '1440',
            'window_res_y': '900',
            'rendering_quality': 'high',
        }.items())]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('seed', default_value='0',
                              description='layout seed (same seed = same arena)'),
        OpaqueFunction(function=setup),
    ])
