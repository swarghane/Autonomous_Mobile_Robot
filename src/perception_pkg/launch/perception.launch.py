from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='perception_pkg',
            executable='camera_node',
            name='camera_node',
        ),

        Node(
            package='perception_pkg',
            executable='detector_node',
            name='detector_node',
            # parameters=[{'model_path': 'yolov8n.pt'}],
        ),

        Node(
            package='perception_pkg',
            executable='tracking_node',
            name='tracking_node',
        ),

        Node(
            package='perception_pkg',
            executable='obstacle_detection_node',
            name='obstacle_detection_node',
        ),

    ])
