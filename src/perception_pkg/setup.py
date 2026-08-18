from setuptools import find_packages, setup

package_name = 'perception_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
         ['launch/perception.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sourabh Warghane',
    maintainer_email='sourabhwarghane@gmail.com',
    description='ROS 2 perception package for camera processing, TensorRT-based object detection, object tracking, and LiDAR obstacle perception.',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "camera_node = perception_pkg.camera_node:main",
            "detector_node = perception_pkg.detector_node:main",
            "tracking_node = perception_pkg.tracking_node:main",
            "obstacle_detection_node = perception_pkg.obstacle_detection_node:main",
        ],
    },
)
