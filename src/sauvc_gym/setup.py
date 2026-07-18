from setuptools import find_packages, setup

package_name = "sauvc_gym"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test", "test.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/station_keeping.yaml"]),
    ],
    # gymnasium is intentionally NOT in install_requires: it is not a rosdep key
    # and pinning it here fights colcon. Install it into the same interpreter
    # ROS 2 uses:  python3 -m pip install "gymnasium>=1.0" --user
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Mohammad Haadi Akhter",
    maintainer_email="you@example.com",
    description="Gymnasium interface to the SAUVC 2026 Stonefish arena.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "verify_allocation = sauvc_gym.scripts.verify_allocation:main",
            "identify_allocation = sauvc_gym.scripts.identify_allocation:main",
            "random_agent = sauvc_gym.scripts.random_agent:main",
            "train_sb3 = sauvc_gym.scripts.train_sb3:main",
        ],
    },
)
