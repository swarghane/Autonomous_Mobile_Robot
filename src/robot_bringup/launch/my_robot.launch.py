import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    perception_pkg_dir = get_package_share_directory('perception_pkg')
    interaction_pkg_dir = get_package_share_directory('interaction_pkg')
    control_pkg_dir = get_package_share_directory('control_pkg')
    rplidar_ros_dir = get_package_share_directory('rplidar_ros')

    included_perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(perception_pkg_dir, 'launch', 'perception.launch.py')
        )
    )

    included_control_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(control_pkg_dir, 'launch', 'control.launch.py')
        )
    )

    included_interaction_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(interaction_pkg_dir, 'launch', 'interaction.launch.py')
        )
    )

    included_rplidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(rplidar_ros_dir, 'launch', 'rplidar_c1_launch.py')
        )
    )


    return LaunchDescription([

        included_perception_launch,

        included_rplidar_launch,

        included_interaction_launch,

        included_control_launch,


        Node(
            package='rosbridge_server',
            executable='rosbridge_websocket',
            name='rosbridge_websocket',
        ),

        Node(
            package='web_video_server',
            executable='web_video_server',
            name='web_video_server',
            output='screen'
        ),

    ])