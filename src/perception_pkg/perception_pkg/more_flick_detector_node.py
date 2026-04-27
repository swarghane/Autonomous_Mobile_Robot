import time
import cv2
import numpy as np
import rclpy

from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from ultralytics import YOLO
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose


class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector_node')

        # Parameters
        self.declare_parameter('model_path', '/workspace/src/perception_pkg/perception_pkg/yolov8n.engine')
        self.declare_parameter('conf_threshold', 0.5)
        self.declare_parameter('imgsz', 640)

        self.model_path = self.get_parameter('model_path').value
        self.conf_threshold = float(self.get_parameter('conf_threshold').value)
        self.imgsz = int(self.get_parameter('imgsz').value)

        self.bridge = CvBridge()

        # Load TensorRT model
        self.model = YOLO(self.model_path, task='detect')

        # Warmup (IMPORTANT)
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model(dummy, imgsz=self.imgsz, verbose=False)

        self.get_logger().info(
            f'Model loaded: {self.model_path} | imgsz={self.imgsz} | conf={self.conf_threshold}'
        )

        # Performance control
        self.processing = False
        self.frame_count = 0
        self.last_stats_time = time.perf_counter()

        # Subscriber (camera)
        self.sub_ = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            qos_profile_sensor_data
        )

        # Publisher (detections)
        self.pub_ = self.create_publisher(
            Detection2DArray,
            '/detections',
            qos_profile_sensor_data
        )

        self.get_logger().info('🚀 YOLO Perception Node Started (TensorRT)')

    def image_callback(self, msg: Image):

        # Drop frame if still processing
        if self.processing:
            return

        self.processing = True
        start_time = time.perf_counter()

        try:
            # Convert ROS → OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            h, w = frame.shape[:2]

            # Resize for inference (important for speed)
            resized = cv2.resize(frame, (self.imgsz, self.imgsz))

            # Run inference (TensorRT auto GPU)
            results = self.model(
                resized,
                imgsz=self.imgsz,
                verbose=False
            )

            detection_array = Detection2DArray()
            detection_array.header = msg.header

            detection_count = 0

            scale_x = w / self.imgsz
            scale_y = h / self.imgsz

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

                    # Scale back to original image
                    cx *= scale_x
                    cy *= scale_y
                    bw *= scale_x
                    bh *= scale_y

                    detection = Detection2D()
                    detection.header = msg.header

                    # ✅ Correct fields
                    detection.bbox.center.position.x = float(cx)
                    detection.bbox.center.position.y = float(cy)
                    detection.bbox.size_x = float(bw)
                    detection.bbox.size_y = float(bh)

                    hypothesis = ObjectHypothesisWithPose()
                    hypothesis.hypothesis.class_id = class_name
                    hypothesis.hypothesis.score = confidence

                    detection.results.append(hypothesis)
                    detection_array.detections.append(detection)

                    detection_count += 1

            # Publish detections
            self.pub_.publish(detection_array)

            # Performance logging
            self.frame_count += 1
            elapsed = time.perf_counter() - start_time

            if self.frame_count % 30 == 0:
                now = time.perf_counter()
                window = now - self.last_stats_time
                fps = 30.0 / window if window > 0 else 0.0
                self.last_stats_time = now

                self.get_logger().info(
                    f'⚡ FPS={fps:.2f} | '
                    f'Latency={elapsed*1000:.1f} ms | '
                    f'Detections={detection_count}'
                )

        except Exception as e:
            self.get_logger().error(f'❌ Perception failed: {e}')

        finally:
            self.processing = False

    def destroy_node(self):
        self.get_logger().info('Shutting down YOLO node.')
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = DetectorNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # This catches the Ctrl+C gracefully
        node.get_logger().info('Detector node stopping...')
    finally:
        # Check if it's still active before shutting down
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == "__main__":
    main()