import os
from glob import glob

from setuptools import setup

package_name = 'sauvc_traj_recorder'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name, package_name + '.trajectory_tools'],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='sauvc_team',
    maintainer_email='you@example.com',
    description=(
        'Stonefish-only trajectory recorder: raw camera topics + ground-truth '
        'odometry to rosbag2 + image/CSV logs, one folder per trajectory.'
    ),
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'trajectory_recorder_node = '
            'sauvc_traj_recorder.trajectory_recorder_node:main',
            'csv_to_tum = '
            'sauvc_traj_recorder.trajectory_tools.csv_to_tum:main',
            'align_and_evaluate = '
            'sauvc_traj_recorder.trajectory_tools.align_and_evaluate:main',
        ],
    },
)
