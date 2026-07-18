import os
from glob import glob
from setuptools import setup

package_name = 'sauvc_sim_bridge'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='SAUVC Team',
    maintainer_email='team@sauvc.local',
    description='Stonefish -> real-stack shim layer and sim-only diagnostics.',
    license='MIT',
    entry_points={'console_scripts': [
        'imu_shim_node = sauvc_sim_bridge.imu_shim_node:main',
        'depth_shim_node = sauvc_sim_bridge.depth_shim_node:main',
        'rtf_monitor_node = sauvc_sim_bridge.rtf_monitor_node:main',
        'flow_scorer_node = sauvc_sim_bridge.flow_scorer_node:main',
        'image_relay_node = sauvc_sim_bridge.image_relay_node:main',
        'direct_control_node = sauvc_sim_bridge.direct_control_node:main',
        'ardusub_setpoint_node = sauvc_sim_bridge.ardusub_setpoint_node:main',
    ]},
)
