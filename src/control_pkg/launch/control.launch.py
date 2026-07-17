from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='control_pkg',
            executable='decision_node',
            name='decision_node',
        ),

        Node(
            package='control_pkg',
            executable='motor_control_node',
            name='motor_control_node',
        ),

        # Node(
        #     package='control_pkg',
        #     executable='display_node',
        #     name='display_node',
        # ),
    ])
