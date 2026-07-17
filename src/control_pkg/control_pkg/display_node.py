import rclpy
from rclpy.node import Node
import cv2
from vision_msgs.msg import Detection2D, Detection2DArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
import numpy as np


class DisplayNode(Node):
    def __init__(self):
        super().__init__('display_node')

        self.bridge = CvBridge()
        self.latest_msg = None

        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )
        detection_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        self.image_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, image_qos)

        self.detector_sub = self.create_subscription(
            Detection2DArray, '/tracked_detections', self.detection_callback, detection_qos)

        self.rviz_pub = self.create_publisher(
            Image, '/rviz_debug_image', 10)

    def detection_callback(self, msg):
        self.latest_msg = msg
        # self.latest_msg.append(msg)
        # self.latest_msg = self.latest_msg[-10:]

    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # frame = cv2.flip(frame, 1)

        if self.latest_msg is not None:
            for det in self.latest_msg.detections:
                # 1. vision_msgs uses Center X, Center Y, Width, Height
                cx = det.bbox.center.position.x
                cy = det.bbox.center.position.y
                w = det.bbox.size_x
                h = det.bbox.size_y

                # 2. Convert to Top-Left and Bottom-Right for OpenCV drawing
                x_min = int(cx - (w / 2))
                y_min = int(cy - (h / 2))
                x_max = int(cx + (w / 2))
                y_max = int(cy + (h / 2))

                # 3. Get Class ID and Score from the results list
                # Most detectors put the best match at index 0
                if len(det.results) > 0:
                    class_id = det.results[0].hypothesis.class_id
                    score = det.results[0].hypothesis.score
                    track_id = det.id
                    label = f'{class_id} ID:{track_id} {score:.2f}'
                else:
                    label = "Object"

                # 4. Draw
                cv2.rectangle(frame, (x_min, y_min),
                              (x_max, y_max), (0, 255, 0), 2)
                cv2.putText(frame, label, (x_min, max(20, y_min - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            out_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            out_msg.header = msg.header  # Keep the timestamp same as image
            self.rviz_pub.publish(out_msg)

        cv2.imshow('YOLO/MediaPipe Detections', frame)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = DisplayNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # This catches the Ctrl+C gracefully
        node.get_logger().info('Display node stopping...')
    finally:
        # Check if it's still active before shutting down
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()
