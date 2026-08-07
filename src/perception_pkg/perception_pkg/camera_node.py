import threading
import time

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, CompressedImage, Image
from std_msgs.msg import Empty


def gstreamer_pipeline(
    camera_type="csi",
    sensor_id=0,
    device_id=0,
    capture_width=1920,
    capture_height=1080,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    """Build the GStreamer pipeline for a CSI or USB camera."""
    if camera_type == "csi":
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width={capture_width}, "
            f"height={capture_height}, format=NV12, framerate={framerate}/1 ! "
            f"nvvidconv flip-method={flip_method} ! "
            f"video/x-raw, width={display_width}, height={display_height}, "
            "format=BGRx ! videoconvert ! video/x-raw, format=BGR ! "
            "appsink drop=true sync=false"
        )

    if camera_type == "usb":
        return (
            f"v4l2src device=/dev/video{device_id} ! "
            f"video/x-raw, width={capture_width}, height={capture_height}, "
            f"framerate={framerate}/1 ! "
            "videoconvert ! video/x-raw, format=BGR ! "
            "appsink drop=true sync=false"
        )

    raise ValueError(f"Unsupported camera_type: {camera_type}")


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")

        self.bridge = CvBridge()

        # Camera source and image settings
        self.declare_parameter("camera_type", "csi")
        self.declare_parameter("sensor_id", 0)
        self.declare_parameter("device_id", 0)
        self.declare_parameter("flip_method", 0)
        self.declare_parameter("capture_width", 1920)
        self.declare_parameter("capture_height", 1080)
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("camera_fps", 30.0)
        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("frame_id", "camera_link")
        self.declare_parameter("jpeg_quality", 80)
        self.declare_parameter("publish_compressed", False)
        self.declare_parameter("publish_camera_info", True)

        # ROS topic and heartbeat settings
        self.declare_parameter("image_raw_topic", "/camera/image_raw")
        self.declare_parameter(
            "image_compressed_topic", "/camera/image_compressed"
        )
        self.declare_parameter("camera_info_topic", "/camera/camera_info")
        self.declare_parameter("camera_heartbeat_topic", "/camera/heartbeat")
        self.declare_parameter("camera_heartbeat_rate", 1.0)

        # Recovery, shutdown, and diagnostic settings
        self.declare_parameter("enable_reconnect", True)
        self.declare_parameter("reconnect_after_failures", 30)
        self.declare_parameter("camera_open_delay_sec", 2.0)
        self.declare_parameter("shutdown_release_timeout_sec", 2.0)
        self.declare_parameter("stats_log_interval_sec", 5.0)

        # Simplified intrinsic camera values
        self.declare_parameter("camera_fx", 600.0)
        self.declare_parameter("camera_fy", 600.0)

        self.camera_type = str(self.get_parameter("camera_type").value)
        self.sensor_id = int(self.get_parameter("sensor_id").value)
        self.device_id = int(self.get_parameter("device_id").value)
        self.flip_method = int(self.get_parameter("flip_method").value)
        self.capture_width = int(self.get_parameter("capture_width").value)
        self.capture_height = int(self.get_parameter("capture_height").value)
        self.image_width = int(self.get_parameter("image_width").value)
        self.image_height = int(self.get_parameter("image_height").value)
        self.camera_fps = float(self.get_parameter("camera_fps").value)
        self.publish_rate = float(self.get_parameter("publish_rate").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.publish_compressed = bool(
            self.get_parameter("publish_compressed").value
        )
        self.publish_camera_info = bool(
            self.get_parameter("publish_camera_info").value
        )
        self.image_raw_topic = str(
            self.get_parameter("image_raw_topic").value
        )
        self.image_compressed_topic = str(
            self.get_parameter("image_compressed_topic").value
        )
        self.camera_info_topic = str(
            self.get_parameter("camera_info_topic").value
        )
        self.camera_heartbeat_topic = str(
            self.get_parameter("camera_heartbeat_topic").value
        )
        self.camera_heartbeat_rate = float(
            self.get_parameter("camera_heartbeat_rate").value
        )
        self.enable_reconnect = bool(
            self.get_parameter("enable_reconnect").value
        )
        self.reconnect_after_failures = int(
            self.get_parameter("reconnect_after_failures").value
        )
        self.camera_open_delay_sec = float(
            self.get_parameter("camera_open_delay_sec").value
        )
        self.shutdown_release_timeout_sec = float(
            self.get_parameter("shutdown_release_timeout_sec").value
        )
        self.stats_log_interval_sec = float(
            self.get_parameter("stats_log_interval_sec").value
        )
        self.camera_fx = float(self.get_parameter("camera_fx").value)
        self.camera_fy = float(self.get_parameter("camera_fy").value)

        self.failed_reads = 0
        self.total_failed_reads = 0
        self.total_frames_published = 0
        self.last_successful_frame_time = None

        # Sensor streams use low-latency QoS so old frames are discarded.
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.image_pub = self.create_publisher(
            Image, self.image_raw_topic, image_qos
        )
        self.compressed_pub = self.create_publisher(
            CompressedImage, self.image_compressed_topic, image_qos
        )
        self.camera_info_pub = self.create_publisher(
            CameraInfo, self.camera_info_topic, image_qos
        )
        self.camera_heartbeat_pub = self.create_publisher(
            Empty, self.camera_heartbeat_topic, 1
        )

        self.cap = None
        self.open_camera()

        self.timer = self.create_timer(
            1.0 / self.publish_rate, self.timer_callback
        )
        self.camera_heartbeat_timer = self.create_timer(
            1.0 / self.camera_heartbeat_rate,
            self.publish_camera_heartbeat,
        )

        self.get_logger().info("✅ Camera node started successfully")

    def open_camera(self):
        """Open the configured camera and use OpenCV capture as a fallback."""
        if self.cap is not None:
            self.cap.release()

        pipeline = gstreamer_pipeline(
            camera_type=self.camera_type,
            sensor_id=self.sensor_id,
            device_id=self.device_id,
            capture_width=self.capture_width,
            capture_height=self.capture_height,
            display_width=self.image_width,
            display_height=self.image_height,
            framerate=int(self.camera_fps),
            flip_method=self.flip_method,
        )
        self.get_logger().info(f"🎥 Pipeline: {pipeline}")

        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        time.sleep(self.camera_open_delay_sec)

        if not self.cap.isOpened():
            self.get_logger().warn(
                "⚠️ GStreamer failed. Falling back to OpenCV..."
            )
            self.cap = cv2.VideoCapture(self.device_id)

        if not self.cap.isOpened():
            raise RuntimeError("❌ Camera could not be opened")

        self.get_logger().info("✅ Camera opened")

    def create_camera_info_msg(self, stamp):
        """Create the basic CameraInfo message associated with a frame."""
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.width = self.image_width
        msg.height = self.image_height
        msg.k = [
            self.camera_fx,
            0.0,
            self.image_width / 2,
            0.0,
            self.camera_fy,
            self.image_height / 2,
            0.0,
            0.0,
            1.0,
        ]
        return msg

    def publish_camera_heartbeat(self):
        """Publish a heartbeat only while frames are being received."""
        if self.last_successful_frame_time is None:
            return

        frame_age = time.monotonic() - self.last_successful_frame_time
        if frame_age <= 2.0:
            self.camera_heartbeat_pub.publish(Empty())

    def timer_callback(self):
        """Read and publish one frame, reconnecting after repeated failures."""
        ret, frame = self.cap.read()

        if not ret:
            self.failed_reads += 1
            self.total_failed_reads += 1

            if (
                self.enable_reconnect
                and self.failed_reads >= self.reconnect_after_failures
            ):
                self.get_logger().warn("🔁 Reconnecting camera...")
                self.open_camera()
                self.failed_reads = 0
            return

        self.failed_reads = 0
        self.last_successful_frame_time = time.monotonic()
        stamp = self.get_clock().now().to_msg()

        img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = self.frame_id
        self.image_pub.publish(img_msg)

        if self.publish_compressed:
            success, encoded = cv2.imencode(
                ".jpg",
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
            )
            if success:
                comp_msg = CompressedImage()
                comp_msg.header.stamp = stamp
                comp_msg.header.frame_id = self.frame_id
                comp_msg.format = "jpeg"
                comp_msg.data = encoded.tobytes()
                self.compressed_pub.publish(comp_msg)

        if self.publish_camera_info:
            self.camera_info_pub.publish(
                self.create_camera_info_msg(stamp)
            )

        self.total_frames_published += 1

    def destroy_node(self):
        """Release the camera without allowing shutdown to block indefinitely."""
        if self.cap:
            cap = self.cap
            self.cap = None

            def _release():
                try:
                    cap.release()
                except Exception as error:
                    self.get_logger().warn(
                        f"Camera release error: {error}"
                    )

            release_thread = threading.Thread(target=_release, daemon=True)
            release_thread.start()
            release_thread.join(timeout=self.shutdown_release_timeout_sec)

        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if rclpy.ok():
            node.get_logger().info("Camera node stopping...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
