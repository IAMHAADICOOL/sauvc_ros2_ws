from setuptools import setup
package_name = 'sauvc_vision'
setup(
    name=package_name, version='0.1.0',
    packages=[package_name],
    data_files=[('share/ament_index/resource_index/packages', ['resource/' + package_name]),
                ('share/' + package_name, ['package.xml'])],
    install_requires=['setuptools'], zip_safe=True,
    entry_points={'console_scripts': [
        'gate_detector_node = sauvc_vision.gate_detector_node:main',
        'image_saver_node = sauvc_vision.image_saver_node:main'
    ]},
)
