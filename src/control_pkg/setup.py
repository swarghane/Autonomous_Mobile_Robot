from setuptools import find_packages, setup

package_name = 'control_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
         ['launch/control.launch.py']),
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
            'decision_node=control_pkg.decision_node:main',
            'decision_node_lidar_camera=control_pkg.decision_node_lidar_camera:main',
            'display_node=control_pkg.display_node:main',
            'motor_control_node=control_pkg.motor_control_node:main',
            'motor_control_node_WO_batterystats=control_pkg.motor_control_node_WO_batterystats:main',
        ],
    },
)
