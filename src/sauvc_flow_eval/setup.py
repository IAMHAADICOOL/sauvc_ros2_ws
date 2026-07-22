import os
from glob import glob
from setuptools import setup

package_name = 'sauvc_flow_eval'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name, package_name + '.estimators',
              package_name + '.scripts'],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='SAUVC Team',
    maintainer_email='team@sauvc.local',
    description='Localization estimator comparison vs Stonefish ground truth.',
    license='MIT',
    entry_points={'console_scripts': [
        'flow_eval_node = sauvc_flow_eval.flow_eval_node:main',
        'landmark_truth_node = sauvc_flow_eval.landmark_truth_node:main', 
    ]},
)
