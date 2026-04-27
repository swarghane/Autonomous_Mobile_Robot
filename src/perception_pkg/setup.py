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
         ['launch/perception_pipeline.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='swarghane',
    maintainer_email='sourabhwarghane@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            "camera_node = perception_pkg.camera_node:main",
            "detector_node = perception_pkg.detector_node:main",
            "tracker_node = perception_pkg.tracker_node:main",
            "display_node = perception_pkg.display_node:main",
        ],
    },
)
