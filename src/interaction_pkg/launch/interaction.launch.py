from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='interaction_pkg',
            executable='stt_node',
            name='stt_node',
        ),

        Node(
            package='interaction_pkg',
            executable='tts_node',
            name='tts_node',
        ),

        Node(
            package='interaction_pkg',
            executable='llm_node',
            name='llm_node',
        ),
    ])
