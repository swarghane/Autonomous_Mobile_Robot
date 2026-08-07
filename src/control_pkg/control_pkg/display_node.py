import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray


class DisplayNode(Node):
    """Draw tracked detections on camera frames and publish a debug image."""

    def __init__(self):
        super().__init__("display_node")

        self.bridge = CvBridge()
        self.latest_detections = None

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )
        detection_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.image_sub = self.create_subscription(
            Image,
            "/camera/image_raw",
            self.image_callback,
            image_qos,
        )
        self.detection_sub = self.create_subscription(
            Detection2DArray,
            "/tracked_detections",
            self.detection_callback,
            detection_qos,
        )
        self.debug_image_pub = self.create_publisher(
            Image,
            "/rviz_debug_image",
            10,
        )

        self.get_logger().info("Display node started")

    def detection_callback(self, msg: Detection2DArray):
        """Store the latest tracked detections for the next image frame."""
        self.latest_detections = msg

    def image_callback(self, msg: Image):
        """Draw the latest detections on an incoming camera frame."""
        frame = self.bridge.imgmsg_to_cv2(
            msg,
            desired_encoding="bgr8",
        )

        if self.latest_detections is not None:
            for detection in self.latest_detections.detections:
                cx = detection.bbox.center.position.x
                cy = detection.bbox.center.position.y
                width = detection.bbox.size_x
                height = detection.bbox.size_y

                x_min = int(cx - width / 2)
                y_min = int(cy - height / 2)
                x_max = int(cx + width / 2)
                y_max = int(cy + height / 2)

                if detection.results:
                    hypothesis = detection.results[0].hypothesis
                    label = (
                        f"{hypothesis.class_id} "
                        f"ID:{detection.id} "
                        f"{hypothesis.score:.2f}"
                    )
                else:
                    label = "Object"

                cv2.rectangle(
                    frame,
                    (x_min, y_min),
                    (x_max, y_max),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    label,
                    (x_min, max(20, y_min - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

            output_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            output_msg.header = msg.header
            self.debug_image_pub.publish(output_msg)

        cv2.imshow("YOLO/MediaPipe Detections", frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = DisplayNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if rclpy.ok():
            node.get_logger().info("Display node stopping...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
