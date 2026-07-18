import os
from glob import glob
from setuptools import setup

package_name = 'sauvc_teleop'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='SAUVC Team',
    maintainer_email='team@sauvc.local',
    description='4-DOF keyboard teleop with depth-hold for the SAUVC AUV sim.',
    license='MIT',
    entry_points={'console_scripts': [
        'keyboard_teleop_node = sauvc_teleop.keyboard_teleop_node:main',
    ]},
)
