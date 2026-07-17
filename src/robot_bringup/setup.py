from setuptools import find_packages, setup

package_name = 'robot_bringup'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch',
         ['launch/my_robot.launch.py']),
        ('share/' + package_name + '/launch',
         ['launch/my_robot_v1.launch.py']),
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
        ],
    },
)
