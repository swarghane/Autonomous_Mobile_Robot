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
        self.declare_parameter('obstacle_clear_threshold', 0.60)
        self.declare_parameter('front_angle_range', 20)
        self.declare_parameter('front_center_angle_deg', 180.0)
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
        self.obstacle_clear_threshold = float(
            self.get_parameter('obstacle_clear_threshold').value
        )
        self.front_angle_range = float(
            self.get_parameter('front_angle_range').value
        )
        self.front_center_angle_deg = float(
            self.get_parameter('front_center_angle_deg').value
        )
        self.left_sector_start = float(
            self.get_parameter('left_sector_start').value
        )
        self.left_sector_end = float(
            self.get_parameter('left_sector_end').value
        )
        self.right_sector_start = float(
            self.get_parameter('right_sector_start').value
        )
        self.right_sector_end = float(
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
        self.obstacle_active = False
        self.avoidance_direction = self.default_direction

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
        front_ranges = []
        left_ranges = []
        right_ranges = []

        front_angle = math.radians(self.front_angle_range)
        front_center = math.radians(self.front_center_angle_deg)

        left_start = math.radians(self.left_sector_start)
        left_end = math.radians(self.left_sector_end)

        right_start = -math.radians(self.right_sector_start)
        right_end = -math.radians(self.right_sector_end)

        for index, distance in enumerate(msg.ranges):
            if distance <= msg.range_min or distance > msg.range_max:
                continue

            if math.isinf(distance) or math.isnan(distance):
                continue

            angle = msg.angle_min + index * msg.angle_increment

            relative_angle = angle - front_center
            relative_angle = math.atan2(
                math.sin(relative_angle),
                math.cos(relative_angle),
            )

            if abs(relative_angle) <= front_angle:
                front_ranges.append(distance)

            if left_start <= relative_angle <= left_end:
                left_ranges.append(distance)

            if right_start <= relative_angle <= right_end:
                right_ranges.append(distance)

        if not front_ranges:
            self.get_logger().warn('No valid front LiDAR data')
            self.publish_obstacle_status(True, 0.0)
            return

        min_front_distance = min(front_ranges)

        left_avg = (
            sum(left_ranges) / len(left_ranges)
            if left_ranges else 0.0
        )

        right_avg = (
            sum(right_ranges) / len(right_ranges)
            if right_ranges else 0.0
        )

        difference = abs(left_avg - right_avg)

        if difference < self.direction_deadband:
            candidate_direction = self.default_direction
        elif left_avg > right_avg:
            candidate_direction = 'LEFT'
        else:
            candidate_direction = 'RIGHT'

        if self.obstacle_active:
            obstacle_detected = (
                min_front_distance < self.obstacle_clear_threshold
            )
        else:
            obstacle_detected = (
                min_front_distance < self.obstacle_distance_threshold
            )

        if obstacle_detected and not self.obstacle_active:
            self.obstacle_active = True
            self.avoidance_direction = candidate_direction

            self.get_logger().warn(
                f'Obstacle Detected at {min_front_distance:.2f} m | '
                f'Avoiding {self.avoidance_direction}'
            )

        elif not obstacle_detected and self.obstacle_active:
            self.obstacle_active = False
            self.get_logger().info('Front path clear')

        direction_msg = String()

        if self.obstacle_active:
            direction_msg.data = self.avoidance_direction
        else:
            direction_msg.data = candidate_direction

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

        self.previous_obstacle_state = self.obstacle_active

        self.free_direction_publisher.publish(direction_msg)

        self.publish_obstacle_status(
            self.obstacle_active,
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