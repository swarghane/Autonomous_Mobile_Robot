import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, String


class ObstacleDetectionNode(Node):
    """Detect front obstacles and identify the side with more free space."""

    def __init__(self):
        super().__init__('obstacle_detection_node')

        # ROS topics
        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('obstacle_topic', '/obstacle_detected')
        self.declare_parameter('front_distance_topic', '/front_distance')
        self.declare_parameter('free_direction_topic', '/free_direction')

        # Detection and direction-selection settings
        self.declare_parameter('obstacle_distance_threshold', 0.50)
        self.declare_parameter('front_angle_range', 20)
        self.declare_parameter('left_sector_start', 30)
        self.declare_parameter('left_sector_end', 90)
        self.declare_parameter('right_sector_start', 90)
        self.declare_parameter('right_sector_end', 30)
        self.declare_parameter('direction_deadband', 0.30)
        self.declare_parameter('default_direction', 'LEFT')
        self.declare_parameter('stats_log_interval_sec', 1.0)

        self.scan_topic = str(self.get_parameter('scan_topic').value)
        self.obstacle_topic = str(self.get_parameter('obstacle_topic').value)
        self.front_distance_topic = str(
            self.get_parameter('front_distance_topic').value
        )
        self.free_direction_topic = str(
            self.get_parameter('free_direction_topic').value
        )
        self.obstacle_distance_threshold = float(
            self.get_parameter('obstacle_distance_threshold').value
        )
        self.front_angle_range = int(
            self.get_parameter('front_angle_range').value
        )
        self.left_sector_start = int(
            self.get_parameter('left_sector_start').value
        )
        self.left_sector_end = int(
            self.get_parameter('left_sector_end').value
        )
        self.right_sector_start = int(
            self.get_parameter('right_sector_start').value
        )
        self.right_sector_end = int(
            self.get_parameter('right_sector_end').value
        )
        self.direction_deadband = float(
            self.get_parameter('direction_deadband').value
        )
        self.default_direction = str(
            self.get_parameter('default_direction').value
        ).upper()
        self.stats_log_interval_sec = float(
            self.get_parameter('stats_log_interval_sec').value
        )

        self.last_stats_log_time = 0.0
        self.previous_obstacle_state = False

        self.scan_subscriber = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self.scan_callback,
            10,
        )
        self.obstacle_publisher = self.create_publisher(
            Bool,
            self.obstacle_topic,
            10,
        )
        self.distance_publisher = self.create_publisher(
            Float32,
            self.front_distance_topic,
            10,
        )
        self.free_direction_publisher = self.create_publisher(
            String,
            self.free_direction_topic,
            10,
        )

        self.get_logger().info('Obstacle Detection Node Started')

    def scan_callback(self, msg: LaserScan):
        """Process one LiDAR scan and publish obstacle information."""
        ranges = msg.ranges

        left_sector = ranges[self.left_sector_start:self.left_sector_end]
        right_sector = ranges[
            -self.right_sector_start:-self.right_sector_end
        ]

        left_valid = [
            value
            for value in left_sector
            if not math.isinf(value)
            and not math.isnan(value)
            and value > 0.0
        ]
        right_valid = [
            value
            for value in right_sector
            if not math.isinf(value)
            and not math.isnan(value)
            and value > 0.0
        ]

        front_left = ranges[:self.front_angle_range]
        front_right = ranges[-self.front_angle_range:]
        front_ranges = list(front_left) + list(front_right)
        valid_front_ranges = [
            value
            for value in front_ranges
            if not math.isinf(value)
            and not math.isnan(value)
            and value > 0.0
        ]

        if not valid_front_ranges:
            self.get_logger().warn('No valid LiDAR data')
            self.publish_obstacle_status(True, 0.0)
            return

        min_front_distance = min(valid_front_ranges)
        left_avg = sum(left_valid) / len(left_valid) if left_valid else 0.0
        right_avg = (
            sum(right_valid) / len(right_valid) if right_valid else 0.0
        )

        direction_msg = String()

        # Use a fixed direction near equality to avoid rapid left-right changes.
        difference = abs(left_avg - right_avg)
        if difference < self.direction_deadband:
            direction_msg.data = self.default_direction
        elif left_avg > right_avg:
            direction_msg.data = 'LEFT'
        else:
            direction_msg.data = 'RIGHT'

        current_time = time.monotonic()
        if (
            current_time - self.last_stats_log_time
            >= self.stats_log_interval_sec
        ):
            self.get_logger().info(
                f'Left:{left_avg:.2f}m | '
                f'Right:{right_avg:.2f}m | '
                f'Front:{min_front_distance:.2f}m | '
                f'Direction:{direction_msg.data}'
            )
            self.last_stats_log_time = current_time

        obstacle_detected = (
            min_front_distance < self.obstacle_distance_threshold
        )

        if obstacle_detected != self.previous_obstacle_state:
            if obstacle_detected:
                self.get_logger().warn(
                    f'Obstacle Detected at {min_front_distance:.2f} m'
                )
            else:
                self.get_logger().info('Front path clear')
            self.previous_obstacle_state = obstacle_detected

        self.free_direction_publisher.publish(direction_msg)
        self.publish_obstacle_status(
            obstacle_detected,
            min_front_distance,
        )

    def publish_obstacle_status(self, detected: bool, distance: float):
        """Publish the obstacle flag and minimum front distance."""
        obstacle_msg = Bool()
        obstacle_msg.data = detected

        distance_msg = Float32()
        distance_msg.data = float(distance)

        self.obstacle_publisher.publish(obstacle_msg)
        self.distance_publisher.publish(distance_msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleDetectionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if rclpy.ok():
            node.get_logger().info('Obstacle detection node stopping...')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
