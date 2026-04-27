import time

import cv2
import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose


class YoloPerceptionNode(Node):
    def __init__(self):
        super().__init__('yolo_perception_node')

        if torch.cuda.is_available():
            self.device = 'cuda'
            self.get_logger().info(
                f'CUDA-capable device detected. Attempting inference on {self.device}. '
                f'PyTorch CUDA version: {torch.version.cuda}'
            )
        else:
            self.device = 'cpu'
            self.get_logger().warn('CUDA device not available. Running inference on CPU.')

        self.declare_parameter('model_path', 'yolov8n.onnx')
        self.declare_parameter('conf_threshold', 0.6)
        self.declare_parameter('imgsz', 320)

        self.bridge = CvBridge()

        self.model_path = str(self.get_parameter('model_path').value)
        self.conf_threshold = float(self.get_parameter('conf_threshold').value)
        self.imgsz = int(self.get_parameter('imgsz').value)

        self.model = YOLO(self.model_path, task='detect')

        # Warmup
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model(dummy, imgsz=self.imgsz, device=self.device, verbose=False)

        self.get_logger().info(
            f'Model loaded: {self.model_path} | device={self.device} | imgsz={self.imgsz} | '
            f'conf_threshold={self.conf_threshold}'
        )

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

        self.sub_ = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            image_qos
        )

        self.pub_ = self.create_publisher(
            Detection2DArray,
            '/detections',
            detection_qos
        )

        self.get_logger().info('Yolo Perception Node started.')

    def image_callback(self, msg: Image) -> None:
        start_time = time.perf_counter()

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            results = self.model(
                frame,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False
            )

            detection_array = Detection2DArray()
            detection_array.header = msg.header

            detection_count = 0

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    confidence = float(box.conf[0])
                    if confidence < self.conf_threshold:
                        continue

                    class_index = int(box.cls[0])
                    class_name = self.model.names[class_index]

                    xywh = box.xywh[0]

                    detection = Detection2D()
                    detection.header = msg.header
                    detection.bbox.center.position.x = float(xywh[0])
                    detection.bbox.center.position.y = float(xywh[1])
                    detection.bbox.size_x = float(xywh[2])
                    detection.bbox.size_y = float(xywh[3])

                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = class_name
                    hypothesis.hypothesis.score = confidence

                    detection.results.append(hypothesis)
                    detection_array.detections.append(detection)
                    detection_count += 1

            self.pub_.publish(detection_array)

            self.frame_count += 1
            elapsed = time.perf_counter() - start_time

            # Log every 30 frames
            if self.frame_count % 30 == 0:
                now = time.perf_counter()
                window = now - self.last_stats_time
                fps = 30.0 / window if window > 0 else 0.0
                self.last_stats_time = now

                self.get_logger().info(
                    f'Perception stats | fps={fps:.2f} | '
                    f'callback_time={elapsed * 1000:.1f} ms | '
                    f'detections={detection_count}'
                )

        except Exception as e:
            self.get_logger().error(f'Perception callback failed: {e}')

    def destroy_node(self):
        self.get_logger().info('Shutting down Yolo Perception Node.')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = YoloPerceptionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
