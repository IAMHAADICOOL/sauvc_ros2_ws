from setuptools import setup
package_name = 'sauvc_drivers'
setup(
    name=package_name, version='0.1.0',
    packages=[package_name],
    data_files=[('share/ament_index/resource_index/packages', ['resource/' + package_name]),
                ('share/' + package_name, ['package.xml'])],
    install_requires=['setuptools'], zip_safe=True,
    entry_points={'console_scripts': [
        'depth_altitude_node = sauvc_drivers.depth_altitude_node:main',
    ]},
)
