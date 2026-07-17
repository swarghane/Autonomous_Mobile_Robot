import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description():

    perception_pkg_dir = get_package_share_directory('perception_pkg')
    # ai_agent_pkg_dir = get_package_share_directory('ai_agent_pkg')
    rplidar_ros_dir = get_package_share_directory('rplidar_ros')

    included_perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(perception_pkg_dir, 'launch', 'perception.launch.py')
        )
    )

    # included_ai_agent_launch = IncludeLaunchDescription(
    #     PythonLaunchDescriptionSource(
    #         os.path.join(ai_agent_pkg_dir, 'launch', 'ai_agent.launch.py')
    #     )
    # )

    included_rplidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(rplidar_ros_dir, 'launch', 'rplidar_c1_launch.py')
        )
    )


    return LaunchDescription([

        included_perception_launch,

        Node(
            package='tracking_pkg',
            executable='tracking_node',
            name='tracking_node',
            output='screen'
        ),

        included_rplidar_launch,

        Node(
            package='lidar_pkg',
            executable='obstacle_detection_node',
            name='obstacle_detection_node',
            output='screen'
        ),

        # included_ai_agent_launch,

        # Node(
        #     package='audio_pkg',
        #     executable='stt_node',
        #     name='stt_node',
        #     output='screen'
        # ),

        Node(
            package='decision_pkg',
            executable='decision_node_lidar_camera',
            name='decision_node_lidar_camera',
            output='screen'
        ),

        Node(
            package='motor_control_pkg',
            executable='motor_control_node',
            name='motor_control_node',
            output='screen',
            # parameters=['config/motor.yaml'],
        ),

        # Node(
        #     package='display_pkg',
        #     executable='display_node',
        #     name='display_node',
        #     output='screen'
        # ),

    ])