from setuptools import setup
package_name = 'sauvc_sensor_check'
setup(
    name=package_name, version='0.1.0',
    packages=[package_name],
    data_files=[('share/ament_index/resource_index/packages', ['resource/' + package_name]),
                ('share/' + package_name, ['package.xml'])],
    install_requires=['setuptools'], zip_safe=True,
    entry_points={'console_scripts': [
        'pressure_check = sauvc_sensor_check.pressure_check:main',
        'imu_taobotics_check = sauvc_sensor_check.imu_taobotics_check:main',
        'imu_pixhawk_check = sauvc_sensor_check.imu_pixhawk_check:main',
        'camera_check_down = sauvc_sensor_check.camera_check_down:main',
        'camera_check_front = sauvc_sensor_check.camera_check_front:main',
    ]},
)
