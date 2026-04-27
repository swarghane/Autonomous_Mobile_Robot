import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
import cv2
from cv_bridge import CvBridge
from ultralytics import YOLO
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import torch
import numpy as np


class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector_node')

        if torch.cuda.is_available():
            self.get_logger().info(
                f'SUCCESS: MX330 GPU detected. CUDA Version: {torch.version.cuda}')
            self.device = 'cuda'
        else:
            self.get_logger().warn('WARNING: GPU not found. Running on slow CPU mode.')
            self.device = 'cpu'

        # self.declare_parameter('model_path', '/workspace/src/perception_pkg/perception_pkg/yolov8n.engine')
        self.declare_parameter('conf_threshold', 0.6)

        self.bridge = CvBridge()
        # The below code is for only one time
        # self.model = YOLO(self.get_parameter('model_path').value)
        # self.model.export(format="onnx", imgsz=320, simplify=True)
        # self.model.to(self.device)

        self.model = YOLO('/workspace/src/perception_pkg/perception_pkg/yolov8n.engine', task='detect')
        import numpy as np
        self.model(np.zeros((320, 320, 3), dtype=np.uint8), device=self.device)

        self.get_logger().info(f'ONNX Model loaded on {self.device}')

        # self.alpha = 0.7
        # self.smoothed_value = None

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
            durability=DurabilityPolicy.VOLATILE,
        )

        self.sub_ = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, image_qos)

        self.pub_ = self.create_publisher(
            Detection2DArray, '/detections', detection_qos)

        self.get_logger().info('Yolo Perception Node started.')

    def image_callback(self, msg):

        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        # frame = cv2.flip(frame, 1)

        # if self.smoothed_value is None:
        #     # First frame: nothing to average with yet
        #     self.smoothed_value = frame.astype(float)
        # else:
        #     # EMA Formula: New = (Alpha * Current) + ((1 - Alpha) * Previous)
        #     self.smoothed_value = (
        #         self.alpha * frame.astype(float)) + ((1 - self.alpha) * self.smoothed_value)

        # processed_frame = self.smoothed_value.astype('uint8')

        results = self.model(
            frame, imgsz=640, device=self.device, verbose=False)

        detection_array = Detection2DArray()
        detection_array.header = msg.header

        for result in results:
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence < self.get_parameter('conf_threshold').value:
                    continue
                class_index = int(box.cls[0])
                class_name = self.model.names[class_index]

                detection = Detection2D()
                xywh = box.xywh[0]
                detection.bbox.center.position.x = float(xywh[0])
                detection.bbox.center.position.y = float(xywh[1])
                detection.bbox.size_x = float(xywh[2])
                detection.bbox.size_y = float(xywh[3])

                hypothesis = ObjectHypothesisWithPose()
                hypothesis.hypothesis.class_id = class_name
                hypothesis.hypothesis.score = confidence

                detection.results.append(hypothesis)
                detection_array.detections.append(detection)

        self.pub_.publish(detection_array)


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


# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
# import cv2
# from cv_bridge import CvBridge
# from ultralytics import YOLO
# import torch
# import os


# class YoloPerceptionNode(Node):
#     def __init__(self):
#         super().__init__('yolo_perception_node')

#         # 1. Setup Device
#         self.device = 0 if torch.cuda.is_available() else 'cpu'
#         self.get_logger().info(f'Using device: {self.device}')

#         # 2. Parameters
#         self.declare_parameter('model_path', 'yolov8n.pt')
#         self.declare_parameter('conf_threshold', 0.5)
#         model_path = self.get_parameter('model_path').value

#         # 3. Handle TensorRT Export (Do this once, not in callback)
#         engine_path = model_path.replace('.pt', '.engine')
#         if not os.path.exists(engine_path):
#             self.get_logger().info(
#                 'Exporting model to TensorRT engine (this may take a few minutes)...')
#             base_model = YOLO(model_path)
#             # half=True is crucial for MX330 performance
#             base_model.export(
#                 format="engine", device=self.device, half=True, imgsz=480)

#         # 4. Load optimized model
#         self.model = YOLO(engine_path, task='detect')
#         self.bridge = CvBridge()

#         # Flag to prevent frame buildup (Drop frames if busy)
#         self.is_processing = False

#         # 5. QoS: Best Effort + Depth 1 is key to reducing lag
#         from rclpy.qos import qos_profile_sensor_data
#         self.sub_ = self.create_subscription(
#             Image, '/camera/image_raw', self.image_callback, qos_profile_sensor_data)

#         self.pub_ = self.create_publisher(Detection2DArray, '/detections', 10)
#         self.get_logger().info('YOLOv8 TensorRT Node Ready.')

#     def image_callback(self, msg):
#         # Drop frame if previous inference is still running
#         if self.is_processing:
#             return

#         self.is_processing = True

#         try:
#             # Convert and downscale if necessary to save MX330 resources
#             frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

#             # Inference using the .engine model
#             # We use stream=False here because we process one frame at a time
#             results = self.model.predict(frame, conf=self.get_parameter('conf_threshold').value,
#                                          device=self.device, verbose=False, half=True)

#             detection_array = Detection2DArray()
#             detection_array.header = msg.header

#             for result in results:
#                 for box in result.boxes:
#                     detection = Detection2D()

#                     # Coordinate mapping
#                     xywh = box.xywh[0].cpu().numpy()
#                     detection.bbox.center.position.x = float(xywh[0])
#                     detection.bbox.center.position.y = float(xywh[1])
#                     detection.bbox.size_x = float(xywh[2])
#                     detection.bbox.size_y = float(xywh[3])

#                     hypothesis = ObjectHypothesisWithPose()
#                     hypothesis.hypothesis.class_id = str(int(box.cls[0]))
#                     hypothesis.hypothesis.score = float(box.conf[0])

#                     detection.results.append(hypothesis)
#                     detection_array.detections.append(detection)

#             self.pub_.publish(detection_array)

#         except Exception as e:
#             self.get_logger().error(f'Inference error: {e}')
#         finally:
#             self.is_processing = False


# def main(args=None):
#     rclpy.init(args=args)
#     node = YoloPerceptionNode()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()


# if __name__ == "__main__":
#     main()
