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
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from ultralytics import YOLO
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose


# COCO keypoint indices (standard YOLOv8-pose ordering)
KP_LEFT_SHOULDER  = 5
KP_RIGHT_SHOULDER = 6
KP_LEFT_WRIST     = 9
KP_RIGHT_WRIST    = 10
KP_CONF_MIN       = 0.4   # ignore low-confidence keypoints


class DetectorNode(Node):
    def __init__(self):
        super().__init__('detector_node')

        # -----------------------------
        # Parameters
        # -----------------------------
        # NOTE: swapped to the pose engine — gives person boxes (for
        # tracking/following, unchanged) AND keypoints (for gestures) from
        # one model, instead of running a second model alongside it.
        self.declare_parameter('model_path', '/workspace/models/vision/yolov8n-pose.engine')
        self.declare_parameter('conf_threshold', 0.6)
        self.declare_parameter('imgsz', 640)
        self.declare_parameter('persistence_time', 0.3)

        # Gesture (raised-hand) tuning
        self.declare_parameter('gesture_hold_sec', 1.0)     # how long hand must stay raised
        self.declare_parameter('gesture_cooldown_sec', 5.0) # min gap between re-triggers

        self.model_path = self.get_parameter('model_path').value
        self.conf_threshold = float(self.get_parameter('conf_threshold').value)
        self.imgsz = int(self.get_parameter("imgsz").value)
        self.persistence_time = float(self.get_parameter('persistence_time').value)
        self.gesture_hold_sec = float(self.get_parameter('gesture_hold_sec').value)
        self.gesture_cooldown_sec = float(self.get_parameter('gesture_cooldown_sec').value)

        self.bridge = CvBridge()

        # -----------------------------
        # Load TensorRT model (pose)
        # -----------------------------
        self.model = YOLO(self.model_path, task='pose')

        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.model(dummy, imgsz=self.imgsz, verbose=False)

        self.get_logger().info(f'🚀 Pose model loaded: {self.model_path}')

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
        # Gesture / audio-enable state
        # -----------------------------
        self.audio_enabled = True          # mirrors webpage toggle; default ON
        self.hand_raised_since = None
        self.last_gesture_trigger = 0.0

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

        audio_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL   # late subscribers get last value
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

        # Track the webpage/manual toggle so the gesture only ever flips
        # OFF -> ON, never fights a manual OFF.
        self.audio_state_sub = self.create_subscription(
            Bool, '/audio_enabled', self._audio_state_callback, audio_qos
        )

        # -----------------------------
        # Publishers
        # -----------------------------
        self.pub_ = self.create_publisher(
            Detection2DArray,
            '/detections',
            detection_qos
        )

        self.audio_pub = self.create_publisher(Bool, '/audio_enabled', audio_qos)

        # -----------------------------
        # Worker Thread Initialization
        # -----------------------------
        self.running = True
        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()

        # Stats
        self.frame_count = 0
        self.last_time = time.time()

        self.get_logger().info('✅ YOLO Pose Detector Node Started')

    def _audio_state_callback(self, msg: Bool):
        self.audio_enabled = msg.data

    def image_callback(self, msg):
        if not self.running:
            return
        self.latest_msg = msg
        self.frame_ready_event.set()

    def worker_loop(self):
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

    def _check_raised_hand(self, keypoints_xy, keypoints_conf):
        """Return True if either wrist is clearly above its shoulder."""
        def kp_ok(idx):
            return keypoints_conf[idx] >= KP_CONF_MIN

        raised = False
        if kp_ok(KP_LEFT_WRIST) and kp_ok(KP_LEFT_SHOULDER):
            if keypoints_xy[KP_LEFT_WRIST][1] < keypoints_xy[KP_LEFT_SHOULDER][1]:
                raised = True
        if kp_ok(KP_RIGHT_WRIST) and kp_ok(KP_RIGHT_SHOULDER):
            if keypoints_xy[KP_RIGHT_WRIST][1] < keypoints_xy[KP_RIGHT_SHOULDER][1]:
                raised = True
        return raised

    def _handle_gesture(self, any_hand_raised):
        now = time.time()

        if not any_hand_raised:
            self.hand_raised_since = None
            return

        if self.hand_raised_since is None:
            self.hand_raised_since = now
            return

        held_for = now - self.hand_raised_since
        if held_for < self.gesture_hold_sec:
            return

        # Only ever turns audio ON, and only if currently OFF, and respects
        # a cooldown so a continued raised hand doesn't spam re-triggers.
        if self.audio_enabled:
            return
        if (now - self.last_gesture_trigger) < self.gesture_cooldown_sec:
            return

        self.audio_enabled = True
        self.last_gesture_trigger = now
        self.hand_raised_since = None

        msg = Bool()
        msg.data = True
        self.audio_pub.publish(msg)
        self.get_logger().info('🖐️ Raised-hand gesture detected — audio re-enabled')

    def process_frame(self, msg):
        start_time = time.time()

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            results = self.model(frame, imgsz=self.imgsz, verbose=False)

            detection_array = Detection2DArray()
            detection_array.header = msg.header
            detection_found = False
            any_hand_raised = False

            for result in results:
                if result.boxes is None:
                    continue

                # Keypoints array aligns index-for-index with result.boxes
                kp_xy_all   = result.keypoints.xy.cpu().numpy() if result.keypoints is not None else None
                kp_conf_all = result.keypoints.conf.cpu().numpy() if (result.keypoints is not None and result.keypoints.conf is not None) else None

                for i, box in enumerate(result.boxes):
                    confidence = float(box.conf[0].item())

                    if confidence < self.conf_threshold:
                        continue

                    class_index = int(box.cls[0].item())
                    class_name = self.model.names[class_index]  # 'person' for pose models

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

                    # ── Gesture check for this person ──
                    if kp_xy_all is not None and kp_conf_all is not None and i < len(kp_xy_all):
                        if self._check_raised_hand(kp_xy_all[i], kp_conf_all[i]):
                            any_hand_raised = True

            self._handle_gesture(any_hand_raised)

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

                if rclpy.utilities.ok() and self.running:
                    self.get_logger().info(
                        f'⚡ FPS={fps:.2f} | Latency={latency:.1f} ms | Detections={len(detection_array.detections)}'
                    )

        except Exception as e:
            if rclpy.utilities.ok() and self.running:
                self.get_logger().error(f'❌ Detection error: {e}')

    def destroy_node(self):
        self.running = False
        self.frame_ready_event.set()

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