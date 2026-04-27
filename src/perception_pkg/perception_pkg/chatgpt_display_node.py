import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray


class DisplayNode(Node):
    def __init__(self):
        super().__init__('display_node')

        self.bridge = CvBridge()
        self.latest_msg = None

        self.declare_parameter('show_window', True)
        self.declare_parameter('flip_display', True)
        self.declare_parameter('stale_detection_ms', 200)

        self.show_window = bool(self.get_parameter('show_window').value)
        self.flip_display = bool(self.get_parameter('flip_display').value)
        self.stale_detection_ms = int(
            self.get_parameter('stale_detection_ms').value)

        self.frame_count = 0
        self.last_stats_time = time.perf_counter()

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

        debug_image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            image_qos
        )

        self.detector_sub = self.create_subscription(
            Detection2DArray,
            '/detections',
            self.detection_callback,
            detection_qos
        )

        self.rviz_pub = self.create_publisher(
            Image,
            '/rviz_debug_image',
            debug_image_qos
        )

        self.get_logger().info('Display node started.')

    def detection_callback(self, msg: Detection2DArray) -> None:
        self.latest_msg = msg

    def _header_time_to_ns(self, header) -> int:
        return int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)

    def image_callback(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # if self.flip_display:
            #     frame = cv2.flip(frame, 1)

            frame_h, frame_w = frame.shape[:2]
            image_time_ns = self._header_time_to_ns(msg.header)

            latest_msg = self.latest_msg
            drawn_count = 0

            if latest_msg is not None:
                det_time_ns = self._header_time_to_ns(latest_msg.header)
                age_ms = abs(image_time_ns - det_time_ns) / 1_000_000.0

                if age_ms <= self.stale_detection_ms:
                    for det in latest_msg.detections:
                        cx = det.bbox.center.position.x
                        cy = det.bbox.center.position.y
                        w = det.bbox.size_x
                        h = det.bbox.size_y

                        x_min = max(0, int(cx - (w / 2)))
                        y_min = max(0, int(cy - (h / 2)))
                        x_max = min(frame_w - 1, int(cx + (w / 2)))
                        y_max = min(frame_h - 1, int(cy + (h / 2)))

                        if len(det.results) > 0:
                            class_id = det.results[0].hypothesis.class_id
                            score = det.results[0].hypothesis.score
                            label = f'{class_id}: {score:.2f}'
                        else:
                            label = 'Object'

                        cv2.rectangle(frame, (x_min, y_min),
                                      (x_max, y_max), (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            label,
                            (x_min, max(20, y_min - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2
                        )
                        drawn_count += 1

            out_msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            out_msg.header = msg.header
            self.rviz_pub.publish(out_msg)

            self.frame_count += 1
            if self.frame_count % 30 == 0:
                now = time.perf_counter()
                window = now - self.last_stats_time
                fps = 30.0 / window if window > 0 else 0.0
                self.last_stats_time = now

                self.get_logger().info(
                    f'Display stats | fps={fps:.2f} | drawn_boxes={drawn_count}'
                )

            if self.show_window:
                cv2.imshow('YOLO/MediaPipe Detections', frame)
                cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Display callback failed: {e}')

    def destroy_node(self):
        self.get_logger().info('Shutting down display node.')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DisplayNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
