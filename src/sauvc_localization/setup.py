from setuptools import setup
package_name = 'sauvc_localization'
setup(
    name=package_name, version='0.1.0',
    packages=[package_name],
    data_files=[('share/ament_index/resource_index/packages', ['resource/' + package_name]),
                ('share/' + package_name, ['package.xml'])],
    install_requires=['setuptools'], zip_safe=True,
    entry_points={'console_scripts': [
        'flow_velocity_node = sauvc_localization.flow_velocity_node:main',
        'lane_heading_node = sauvc_localization.lane_heading_node:main',
        'imu_covariance_check = sauvc_localization.imu_covariance_check:main',
        'preint_smoother_node = sauvc_localization.preint_smoother_node:main',
    ]},
)
