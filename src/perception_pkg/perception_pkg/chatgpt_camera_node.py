import time

import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.bridge = CvBridge()

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter("camera_port", 0)
        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("frame_id", "camera_link")
        self.declare_parameter("jpeg_quality", 80)
        self.declare_parameter("publish_compressed", True)
        self.declare_parameter("publish_camera_info", True)

        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("camera_fps", 30.0)
        self.declare_parameter("auto_exposure", 3)

        self.declare_parameter("enable_reconnect", True)
        self.declare_parameter("reconnect_after_failures", 30)
        self.declare_parameter("stats_log_interval_sec", 5.0)
        self.declare_parameter("capture_latency_warn_ms", 50.0)

        self.camera_port = int(self.get_parameter("camera_port").value)
        self.publish_rate = float(self.get_parameter("publish_rate").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.publish_compressed = bool(
            self.get_parameter("publish_compressed").value)
        self.publish_camera_info = bool(
            self.get_parameter("publish_camera_info").value)

        self.image_width = int(self.get_parameter("image_width").value)
        self.image_height = int(self.get_parameter("image_height").value)
        self.camera_fps = float(self.get_parameter("camera_fps").value)
        self.auto_exposure = float(self.get_parameter("auto_exposure").value)

        self.enable_reconnect = bool(
            self.get_parameter("enable_reconnect").value)
        self.reconnect_after_failures = int(
            self.get_parameter("reconnect_after_failures").value
        )
        self.stats_log_interval_sec = float(
            self.get_parameter("stats_log_interval_sec").value
        )
        self.capture_latency_warn_ms = float(
            self.get_parameter("capture_latency_warn_ms").value
        )

        # -----------------------------
        # Runtime counters / stats
        # -----------------------------
        self.failed_reads = 0
        self.total_failed_reads = 0
        self.total_frames_published = 0
        self.total_compressed_published = 0
        self.total_camera_info_published = 0
        self.total_encode_failures = 0
        self.total_reconnect_attempts = 0

        self.stats_window_frame_count = 0
        self.stats_window_start = time.perf_counter()

        # -----------------------------
        # QoS
        # -----------------------------
        camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # -----------------------------
        # Publishers
        # -----------------------------
        self.image_pub = self.create_publisher(
            Image, '/camera/image_raw', camera_qos
        )

        self.compressed_pub = self.create_publisher(
            CompressedImage, '/camera/image_compressed', camera_qos
        )

        self.camera_info_pub = self.create_publisher(
            CameraInfo, '/camera/camera_info', camera_qos
        )

        # -----------------------------
        # Open camera
        # -----------------------------
        self.cap = None
        self.open_camera()

        # -----------------------------
        # Timer
        # -----------------------------
        period = 1.0 / self.publish_rate
        self.timer = self.create_timer(period, self.timer_callback)

        self.get_logger().info(
            "Camera node started | "
            f"camera_port={self.camera_port}, "
            f"publish_rate={self.publish_rate:.2f} Hz, "
            f"frame_id={self.frame_id}, "
            f"jpeg_quality={self.jpeg_quality}, "
            f"publish_compressed={self.publish_compressed}, "
            f"publish_camera_info={self.publish_camera_info}, "
            f"resolution={self.image_width}x{self.image_height}, "
            f"requested_camera_fps={self.camera_fps:.2f}, "
            f"enable_reconnect={self.enable_reconnect}"
        )

    def open_camera(self):
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

        self.cap = cv2.VideoCapture(self.camera_port, cv2.CAP_V4L2)

        # MJPG is often much better for USB webcam + WSL pipelines
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.image_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.image_height)
        self.cap.set(cv2.CAP_PROP_FPS, self.camera_fps)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, self.auto_exposure)

        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera.")
            raise RuntimeError("Camera could not be opened")

        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS))

        self.get_logger().info(
            "Camera opened successfully | "
            f"actual_resolution={self.actual_width}x{self.actual_height}, "
            f"actual_fps={self.actual_fps:.2f}"
        )

    def try_reconnect(self):
        if not self.enable_reconnect:
            return

        self.total_reconnect_attempts += 1
        self.get_logger().warn(
            f"Attempting camera reconnect #{self.total_reconnect_attempts}..."
        )

        try:
            self.open_camera()
            self.failed_reads = 0
            self.get_logger().info("Camera reconnect successful.")
        except Exception as e:
            self.get_logger().error(f"Camera reconnect failed: {e}")

    def create_camera_info_msg(self, stamp):
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id

        msg.width = self.actual_width
        msg.height = self.actual_height

        # Placeholder / uncalibrated values
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        fx = 600.0
        fy = 600.0
        cx = self.actual_width / 2.0
        cy = self.actual_height / 2.0

        msg.k = [
            fx, 0.0, cx,
            0.0, fy, cy,
            0.0, 0.0, 1.0
        ]

        msg.r = [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0
        ]

        msg.p = [
            fx, 0.0, cx, 0.0,
            0.0, fy, cy, 0.0,
            0.0, 0.0, 1.0, 0.0
        ]

        return msg

    def log_periodic_stats(self):
        now = time.perf_counter()
        elapsed = now - self.stats_window_start

        if elapsed < self.stats_log_interval_sec:
            return

        fps = self.stats_window_frame_count / elapsed if elapsed > 0.0 else 0.0

        self.get_logger().info(
            "Camera stats | "
            f"published_fps={fps:.2f}, "
            f"window_frames={self.stats_window_frame_count}, "
            f"failed_reads_total={self.total_failed_reads}, "
            f"compressed_published_total={self.total_compressed_published}, "
            f"camera_info_published_total={self.total_camera_info_published}, "
            f"encode_failures_total={self.total_encode_failures}, "
            f"reconnect_attempts_total={self.total_reconnect_attempts}"
        )

        self.stats_window_start = now
        self.stats_window_frame_count = 0

    def timer_callback(self):
        callback_start = time.perf_counter()

        try:
            capture_start = time.perf_counter()
            ret, frame = self.cap.read()
            capture_latency_ms = (time.perf_counter() - capture_start) * 1000.0

            if capture_latency_ms > self.capture_latency_warn_ms:
                self.get_logger().warn(
                    f"High camera capture latency: {capture_latency_ms:.2f} ms"
                )

            if not ret or frame is None:
                self.failed_reads += 1
                self.total_failed_reads += 1

                if self.failed_reads % 10 == 0:
                    self.get_logger().warn(
                        f"Camera read failed {self.failed_reads} consecutive times "
                        f"(total_failed_reads={self.total_failed_reads})"
                    )

                if self.failed_reads >= self.reconnect_after_failures:
                    self.try_reconnect()

                self.log_periodic_stats()
                return

            # Reset consecutive failure counter after successful read
            self.failed_reads = 0

            stamp = self.get_clock().now().to_msg()

            # 1. Publish raw image
            image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
            image_msg.header.stamp = stamp
            image_msg.header.frame_id = self.frame_id
            self.image_pub.publish(image_msg)
            self.total_frames_published += 1
            self.stats_window_frame_count += 1

            # 2. Publish compressed image
            if self.publish_compressed:
                success, encoded_img = cv2.imencode(
                    '.jpg',
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                )

                if success:
                    compressed_msg = CompressedImage()
                    compressed_msg.header.stamp = stamp
                    compressed_msg.header.frame_id = self.frame_id
                    compressed_msg.format = "jpeg"
                    compressed_msg.data = encoded_img.tobytes()
                    self.compressed_pub.publish(compressed_msg)
                    self.total_compressed_published += 1
                else:
                    self.total_encode_failures += 1
                    self.get_logger().warn(
                        f"Failed to encode compressed image "
                        f"(encode_failures_total={self.total_encode_failures})"
                    )

            # 3. Publish camera info
            if self.publish_camera_info:
                camera_info_msg = self.create_camera_info_msg(stamp)
                self.camera_info_pub.publish(camera_info_msg)
                self.total_camera_info_published += 1

            callback_time_ms = (time.perf_counter() - callback_start) * 1000.0
            if callback_time_ms > (1000.0 / self.publish_rate) * 0.9:
                self.get_logger().warn(
                    f"Camera callback is close to timer budget: "
                    f"{callback_time_ms:.2f} ms"
                )

            self.log_periodic_stats()

        except Exception as e:
            self.get_logger().error(f"Camera callback failed: {e}")

    def destroy_node(self):
        self.get_logger().info(
            "Shutting down camera node | "
            f"total_frames_published={self.total_frames_published}, "
            f"total_failed_reads={self.total_failed_reads}, "
            f"total_compressed_published={self.total_compressed_published}, "
            f"total_camera_info_published={self.total_camera_info_published}, "
            f"total_encode_failures={self.total_encode_failures}, "
            f"total_reconnect_attempts={self.total_reconnect_attempts}"
        )

        if self.cap is not None and self.cap.isOpened():
            self.cap.release()

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
