from setuptools import setup

package_name = 'sauvc_sensor_tests'

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
            'pressure_test = sauvc_sensor_tests.pressure_test:main',
            'imu_test = sauvc_sensor_tests.imu_test:main',
            'camera_test = sauvc_sensor_tests.camera_test:main',
            'pixhawk_imu_test = sauvc_sensor_tests.pixhawk_imu_test:main'
        ],
    },
)
