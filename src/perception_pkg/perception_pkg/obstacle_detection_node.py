import math
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, String


class ObstacleDetectionNode(Node):
    def __init__(self):
        super().__init__('obstacle_detection_node')

        self.obstacle_distance_threshold = 0.50
        self.front_angle_range = 20

        self.scan_subscriber = self.create_subscription(
            LaserScan,
            '/scan',
            self.scan_callback,
            10
        )

        self.obstacle_publisher = self.create_publisher(
            Bool,
            '/obstacle_detected',
            10
        )

        self.distance_publisher = self.create_publisher(
            Float32,
            '/front_distance',
            10
        )

        self.free_direction_publisher = self.create_publisher(
            String,
            '/free_direction',
            10
        )

        self.get_logger().info("Obstacle Detection Node Started")

    def scan_callback(self, msg: LaserScan):
        ranges = msg.ranges

        left_sector = ranges[30:90]
        right_sector = ranges[-90:-30]

        left_valid = [
            r for r in left_sector
            if not math.isinf(r)
            and not math.isnan(r)
            and r > 0.0
        ]

        right_valid = [
            r for r in right_sector
            if not math.isinf(r)
            and not math.isnan(r)
            and r > 0.0
        ]

        front_left = ranges[:self.front_angle_range]
        front_right = ranges[-self.front_angle_range:]

        front_ranges = list(front_left) + list(front_right)

        valid_front_ranges = [
            r for r in front_ranges
            if not math.isinf(r)
            and not math.isnan(r)
            and r > 0.0
        ]

        if len(valid_front_ranges) == 0:
            self.get_logger().warn("No valid LiDAR data")
            self.publish_obstacle_status(True, 0.0)
            return

        min_front_distance = min(valid_front_ranges)

        left_avg = (
            sum(left_valid) / len(left_valid)
            if len(left_valid) > 0 else 0.0
        )

        right_avg = (
            sum(right_valid) / len(right_valid)
            if len(right_valid) > 0 else 0.0
        )

        direction_msg = String()

        # the code was modified to avoid oscillation when the robot is centered between two obstacles.
        difference = abs(left_avg - right_avg)

        if difference < 0.30: 
            direction_msg.data = 'LEFT' # Or 'RIGHT', just stick to one!
        elif left_avg > right_avg:
            direction_msg.data = 'LEFT'
        else:
            direction_msg.data = 'RIGHT'

        self.get_logger().info(
            f'Left:{left_avg:.2f} Right:{right_avg:.2f} Direction:{direction_msg.data}'
        )

        self.get_logger().info(
            f"Front Distance: {min_front_distance:.2f} m"
        )

        obstacle_detected = (
            min_front_distance <
            self.obstacle_distance_threshold
        )

        if obstacle_detected:
            self.get_logger().warn(
                f"Obstacle Detected at "
                f"{min_front_distance:.2f} m"
            )

        self.publish_obstacle_status(
            obstacle_detected,
            min_front_distance
        )

    def publish_obstacle_status(self, detected, distance):
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
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()