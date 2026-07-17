# This code uses Multi-Threaded Executor (MTE) and a dedicated worker thread to process frames as soon as they arrive, maximizing throughput and minimizing latency. 
# It also includes robust shutdown handling to ensure clean exit without hanging threads.

import time
import threading
import cv2
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.qos import QoSProfile, HistoryPolicy, ReliabilityPolicy, DurabilityPolicy
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
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
        self.declare_parameter('persistence_time', 0.3)

        self.model_path = self.get_parameter('model_path').value
        self.conf_threshold = float(self.get_parameter('conf_threshold').value)
        self.imgsz = int(self.get_parameter("imgsz").value)
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
        # Threading & Events
        # -----------------------------
        self.callback_group = ReentrantCallbackGroup()
        self.latest_msg = None
        self.frame_ready_event = threading.Event()

        # -----------------------------
        # Persistence (anti-flicker)
        # -----------------------------
        self.last_detections = None
        self.last_detection_time = 0
        
        # -----------------------------
        # QoS Profiles
        # -----------------------------
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

        # -----------------------------
        # Subscriber (Raw Image Setup)
        # -----------------------------
        self.sub_ = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            image_qos,
            callback_group=self.callback_group
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
        # Worker Thread Initialization
        # -----------------------------
        self.running = True
        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()

        # Stats
        self.frame_count = 0
        self.last_time = time.time()

        self.get_logger().info('✅ YOLO Detector Node Started')

    def image_callback(self, msg):
        if not self.running:
            return
        self.latest_msg = msg
        self.frame_ready_event.set()

    def worker_loop(self):
        # We also check rclpy.utilities.ok() directly to ensure the context is valid
        while rclpy.utilities.ok() and self.running:
            if not self.frame_ready_event.wait(timeout=0.1):
                continue
            
            self.frame_ready_event.clear()

            if not self.running or not rclpy.utilities.ok():
                break

            if self.latest_msg is None:
                continue

            msg = self.latest_msg
            self.latest_msg = None

            self.process_frame(msg)

    def process_frame(self, msg):
        start_time = time.time()

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            
            # Direct pass to model (Resizing skipped; handled by GStreamer camera node)
            results = self.model(frame, imgsz=self.imgsz, verbose=False)

            detection_array = Detection2DArray()
            detection_array.header = msg.header
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

            # Anti-flicker persistence
            current_time = time.time()
            if detection_found:
                self.last_detections = detection_array
                self.last_detection_time = current_time
            else:
                if (current_time - self.last_detection_time) < self.persistence_time:
                    detection_array = self.last_detections

            # Publish if context remains healthy
            if detection_array is not None and rclpy.utilities.ok() and self.running:
                self.pub_.publish(detection_array)

            # Performance logging
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                now = time.time()
                fps = 30 / (now - self.last_time)
                self.last_time = now
                latency = (time.time() - start_time) * 1000

                # CRITICAL: Double-check context state right before invoking ROS logs
                if rclpy.utilities.ok() and self.running:
                    self.get_logger().info(
                        f'⚡ FPS={fps:.2f} | Latency={latency:.1f} ms | Detections={len(detection_array.detections)}'
                    )

        except Exception as e:
            # Safely capture log errors only if ROS is completely up
            if rclpy.utilities.ok() and self.running:
                self.get_logger().error(f'❌ Detection error: {e}')

    def destroy_node(self):
        # 1. Turn off our internal loop execution immediately
        self.running = False
        
        # 2. Free any thread lingering on a wait condition
        self.frame_ready_event.set()
        
        # 3. Synchronize background worker cleanup
        if hasattr(self, 'worker_thread') and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=0.5)
            
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass 
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()