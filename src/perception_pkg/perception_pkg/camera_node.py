import time
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CompressedImage, CameraInfo
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy


# ==============================
# GStreamer Pipeline Builder
# ==============================
def gstreamer_pipeline(
    camera_type="csi",
    sensor_id=0,
    device_id=0,
    capture_width=1920,        #1280,
    capture_height=1080,       #720,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    if camera_type == "csi":
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, "
            f"format=NV12, framerate={framerate}/1 ! "
            f"nvvidconv flip-method={flip_method} ! "
            f"video/x-raw, width={display_width}, height={display_height}, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false"
        )

    elif camera_type == "usb":
        return (
            f"v4l2src device=/dev/video{device_id} ! "
            f"video/x-raw, width={capture_width}, height={capture_height}, "
            f"framerate={framerate}/1 ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink drop=true sync=false"
        )

    else:
        raise ValueError(f"Unsupported camera_type: {camera_type}")


# ==============================
# Camera Node
# ==============================
class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.bridge = CvBridge()

        # -----------------------------
        # Parameters
        # -----------------------------
        self.declare_parameter("camera_type", "csi")
        self.declare_parameter("sensor_id", 0)
        self.declare_parameter("device_id", 0)
        self.declare_parameter("flip_method", 0)

        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("frame_id", "camera_link")
        self.declare_parameter("jpeg_quality", 80)
        self.declare_parameter("publish_compressed", False)
        self.declare_parameter("publish_camera_info", True)

        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("camera_fps", 30.0)

        self.declare_parameter("enable_reconnect", True)
        self.declare_parameter("reconnect_after_failures", 30)
        self.declare_parameter("stats_log_interval_sec", 5.0)

        # -----------------------------
        # Get parameters
        # -----------------------------
        self.camera_type = self.get_parameter("camera_type").value
        self.sensor_id = int(self.get_parameter("sensor_id").value)
        self.device_id = int(self.get_parameter("device_id").value)
        self.flip_method = int(self.get_parameter("flip_method").value)

        self.publish_rate = float(self.get_parameter("publish_rate").value)
        self.frame_id = str(self.get_parameter("frame_id").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.publish_compressed = bool(self.get_parameter("publish_compressed").value)
        self.publish_camera_info = bool(self.get_parameter("publish_camera_info").value)

        self.image_width = int(self.get_parameter("image_width").value)
        self.image_height = int(self.get_parameter("image_height").value)
        self.camera_fps = float(self.get_parameter("camera_fps").value)

        self.enable_reconnect = bool(self.get_parameter("enable_reconnect").value)
        self.reconnect_after_failures = int(self.get_parameter("reconnect_after_failures").value)
        self.stats_log_interval_sec = float(self.get_parameter("stats_log_interval_sec").value)

        # -----------------------------
        # Stats
        # -----------------------------
        self.failed_reads = 0
        self.total_failed_reads = 0
        self.total_frames_published = 0

        # -----------------------------
        # QoS
        # -----------------------------
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # -----------------------------
        # Publishers
        # -----------------------------
        self.image_pub = self.create_publisher(Image, '/camera/image_raw', qos)
        self.compressed_pub = self.create_publisher(CompressedImage, '/camera/image_compressed', qos)
        self.camera_info_pub = self.create_publisher(CameraInfo, '/camera/camera_info', qos)

        # -----------------------------
        # Open Camera
        # -----------------------------
        self.cap = None
        self.open_camera()

        # -----------------------------
        # Timer
        # -----------------------------
        self.timer = self.create_timer(1.0 / self.publish_rate, self.timer_callback)

        self.get_logger().info("✅ Camera node started successfully")

    # ==============================
    # Open Camera
    # ==============================
    def open_camera(self):
        if self.cap is not None:
            self.cap.release()

        pipeline = gstreamer_pipeline(
            camera_type=self.camera_type,
            sensor_id=self.sensor_id,
            device_id=self.device_id,
            capture_width=1920,
            capture_height=1080,
            display_width=self.image_width,
            display_height=self.image_height,
            framerate=int(self.camera_fps),
            flip_method=self.flip_method
        )

        self.get_logger().info(f"🎥 Pipeline: {pipeline}")

        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        time.sleep(2)

        # Fallback
        if not self.cap.isOpened():
            self.get_logger().warn("⚠️ GStreamer failed. Falling back to OpenCV...")
            self.cap = cv2.VideoCapture(self.device_id)

        if not self.cap.isOpened():
            raise RuntimeError("❌ Camera could not be opened")

        self.get_logger().info("✅ Camera opened")

    # ==============================
    # Camera Info
    # ==============================
    def create_camera_info_msg(self, stamp):
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id
        msg.width = self.image_width
        msg.height = self.image_height

        msg.k = [600.0, 0.0, self.image_width/2,
                 0.0, 600.0, self.image_height/2,
                 0.0, 0.0, 1.0]

        return msg

    # ==============================
    # Timer Callback
    # ==============================
    def timer_callback(self):
        ret, frame = self.cap.read()

        if not ret:
            self.failed_reads += 1
            self.total_failed_reads += 1

            if self.failed_reads >= self.reconnect_after_failures:
                self.get_logger().warn("🔁 Reconnecting camera...")
                self.open_camera()
                self.failed_reads = 0
            return

        self.failed_reads = 0

        stamp = self.get_clock().now().to_msg()

        # Raw Image
        img_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = self.frame_id
        self.image_pub.publish(img_msg)

        # Compressed
        if self.publish_compressed:
            success, encoded = cv2.imencode('.jpg', frame,
                                            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if success:
                comp_msg = CompressedImage()
                comp_msg.header.stamp = stamp
                comp_msg.format = "jpeg"
                comp_msg.data = encoded.tobytes()
                self.compressed_pub.publish(comp_msg)

        # Camera Info
        if self.publish_camera_info:
            self.camera_info_pub.publish(self.create_camera_info_msg(stamp))

        self.total_frames_published += 1

    # ==============================
    # Shutdown
    # ==============================
    def destroy_node(self):
        if self.cap:
            self.cap.release()
        super().destroy_node()


# ==============================
# Main
# ==============================

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        # This catches the Ctrl+C gracefully
        node.get_logger().info('Camera node stopping...')
    finally:
        # Check if it's still active before shutting down
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()