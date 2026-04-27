import time
import cv2
import numpy as np
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose


class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector_node')

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter('model_path', '/workspace/src/perception_pkg/perception_pkg/yolov8n.engine')
        self.declare_parameter('conf_threshold', 0.6)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('persistence_time', 0.3)  # seconds

        self.model_path = self.get_parameter('model_path').value
        self.conf_threshold = float(self.get_parameter('conf_threshold').value)
        self.imgsz = int(self.get_parameter('imgsz').value)
        self.persistence_time = float(self.get_parameter('persistence_time').value)

        self.bridge = CvBridge()

        # -----------------------------
        # Load TensorRT model
        # -----------------------------
        self.model = YOLO(self.model_path, task='detect')

        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model(dummy, imgsz=self.imgsz, verbose=False)

        self.get_logger().info(f'🚀 Model loaded: {self.model_path}')

        # -----------------------------
        # Frame handling (IMPORTANT)
        # -----------------------------
        self.latest_msg = None

        # -----------------------------
        # Persistence (anti-flicker)
        # -----------------------------
        self.last_detections = None
        self.last_detection_time = 0
        
        # -----------------------------
        # QoS Profiles
        # -----------------------------

        image_qos = QoSProfile(
            history = HistoryPolicy.KEEP_LAST,
            depth = 1,
            reliability = ReliabilityPolicy.BEST_EFFORT,
            durability = DurabilityPolicy.VOLATILE
        )

        detection_qos = QoSProfile(
            history = HistoryPolicy.KEEP_LAST,
            depth = 10,
            reliability = ReliabilityPolicy.RELIABLE,
            durability = DurabilityPolicy.VOLATILE
        )


        # -----------------------------
        # Subscriber
        # -----------------------------
        self.sub_ = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            image_qos
        )

        # -----------------------------
        # Publisher
        # -----------------------------
        self.pub_ = self.create_publisher(
            Detection2DArray,
            '/detections',
            detection_qos
        )

        # -----------------------------
        # Timer (process latest frame)
        # -----------------------------
        self.timer = self.create_timer(0.03, self.process_frame)  # ~30 FPS

        # Stats
        self.frame_count = 0
        self.last_time = time.time()

        self.get_logger().info('✅ Stable YOLO Detector Node Started')

    # -----------------------------
    # Store latest frame only
    # -----------------------------
    def image_callback(self, msg):
        self.latest_msg = msg

    # -----------------------------
    # Process latest frame
    # -----------------------------
    def process_frame(self):

        if self.latest_msg is None:
            return

        msg = self.latest_msg
        self.latest_msg = None

        start_time = time.time()

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            h, w = frame.shape[:2]

            # Resize for inference
            resized = cv2.resize(frame, (self.imgsz, self.imgsz))

            results = self.model(resized, imgsz=self.imgsz, verbose=False)

            detection_array = Detection2DArray()
            detection_array.header = msg.header

            scale_x = w / self.imgsz
            scale_y = h / self.imgsz

            detection_found = False

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:
                    confidence = float(box.conf[0].item())

                    if confidence < self.conf_threshold:
                        continue

                    class_index = int(box.cls[0].item())
                    class_name = self.model.names[class_index]

                    xywh = box.xywh[0].cpu().numpy()
                    cx, cy, bw, bh = xywh

                    # Scale back
                    cx *= scale_x
                    cy *= scale_y
                    bw *= scale_x
                    bh *= scale_y

                    detection = Detection2D()
                    detection.header = msg.header

                    detection.bbox.center.position.x = float(cx)
                    detection.bbox.center.position.y = float(cy)
                    detection.bbox.size_x = float(bw)
                    detection.bbox.size_y = float(bh)

                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = class_name
                    hypothesis.hypothesis.score = confidence

                    detection.results.append(hypothesis)
                    detection_array.detections.append(detection)

                    detection_found = True

            # -----------------------------
            # Anti-flicker persistence
            # -----------------------------
            current_time = time.time()

            if detection_found:
                self.last_detections = detection_array
                self.last_detection_time = current_time
            else:
                if (current_time - self.last_detection_time) < self.persistence_time:
                    detection_array = self.last_detections

            # Publish
            if detection_array is not None:
                self.pub_.publish(detection_array)

            # -----------------------------
            # Performance logging
            # -----------------------------
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                now = time.time()
                fps = 30 / (now - self.last_time)
                self.last_time = now

                latency = (time.time() - start_time) * 1000

                self.get_logger().info(
                    f'⚡ FPS={fps:.2f} | Latency={latency:.1f} ms | Detections={len(detection_array.detections)}'
                )

        except Exception as e:
            self.get_logger().error(f'❌ Detection error: {e}')

    def destroy_node(self):
        self.get_logger().info('Shutting down detector node.')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Detector node stopping...')
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()