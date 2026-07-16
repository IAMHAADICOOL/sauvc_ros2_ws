from setuptools import setup

package_name = 'sauvc_ardusub_demo'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Your Name',
    maintainer_email='you@example.com',
    description='SAUVC helper package',
    license='MIT',
    entry_points={
        'console_scripts': [
            'ardusub_mission = sauvc_ardusub_demo.ardusub_mission:main',
            'motor_map_check = sauvc_ardusub_demo.motor_map_check:main'
        ],
    },
)
