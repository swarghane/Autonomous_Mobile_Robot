import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage, CameraInfo
import cv2
from cv_bridge import CvBridge
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import time

def gstreamer_pipeline(
    sensor_id=0,
    capture_width=3280,
    capture_height=2464,
    display_width=640,
    display_height=480,
    framerate=21,
    flip_method=0,
):
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! video/x-raw, format=(string)BGR ! appsink drop=true"
    )

class CameraNode(Node):
    def __init__(self):
        super().__init__('camera_node')

        self.bridge = CvBridge()
        self.failed_reads = 0

        pipeline = gstreamer_pipeline()
        self.get_logger().info(f"GStreamer Pipeline: {pipeline}")

        # self.declare_parameter("camera_port", gstreamer_pipeline())
        self.declare_parameter("publish_rate", 30.0)
        self.declare_parameter("frame_id", "camera_link")
        self.declare_parameter("jpeg_quality", 80)
        self.declare_parameter("publish_compressed", True)
        self.declare_parameter("publish_camera_info", True)

        # camera_port = self.get_parameter("camera_port").value
        publish_rate = float(self.get_parameter("publish_rate").value)
        self.frame_id = self.get_parameter("frame_id").value
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.publish_compressed = bool(
            self.get_parameter("publish_compressed").value)
        self.publish_camera_info = bool(
            self.get_parameter("publish_camera_info").value)

        # Open camera using V4L2 backend
        self.cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        time.sleep(2)

        # MJPG often works better in WSL/webcam setups
        # self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

        # Set requested resolution/FPS
        # self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        # self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        # self.cap.set(cv2.CAP_PROP_FPS, 30)
        # self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)

        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera!")
            raise RuntimeError("Camera could not be opened")

        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS))

        self.get_logger().info(
            f"Camera opened successfully: "
            f"{self.actual_width}x{self.actual_height}, fps={self.actual_fps}"
        )
        self.get_logger().info(
            f"publish_rate={publish_rate}, "
            f"publish_compressed={self.publish_compressed}, "
            f"publish_camera_info={self.publish_camera_info}"
        )

        camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        # Raw image publisher
        self.image_pub = self.create_publisher(
            Image, '/camera/image_raw', camera_qos
        )

        # Compressed image publisher
        self.compressed_pub = self.create_publisher(
            CompressedImage, '/camera/image_compressed', camera_qos
        )

        # Camera info publisher
        self.camera_info_pub = self.create_publisher(
            CameraInfo, '/camera/camera_info', camera_qos
        )

        period = 1.0 / publish_rate
        self.timer = self.create_timer(period, self.timer_callback)

    def create_camera_info_msg(self, stamp):
        """
        Create a basic CameraInfo message.

        This is a placeholder/un-calibrated camera_info.
        Later, when you calibrate the camera, replace K, D, R, P
        with real calibration values.
        """
        msg = CameraInfo()
        msg.header.stamp = stamp
        msg.header.frame_id = self.frame_id

        msg.width = self.actual_width
        msg.height = self.actual_height

        # Placeholder: uncalibrated camera
        msg.distortion_model = "plumb_bob"
        msg.d = [0.0, 0.0, 0.0, 0.0, 0.0]

        # Basic placeholder intrinsics
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

    def timer_callback(self):
        start = time.time()
        ret, frame = self.cap.read()
        capture_latency = time.time() - start

        if capture_latency > 0.05:
            self.get_logger().warn(
                f"Capture latency: {capture_latency * 1000:.2f} ms"
            )

        if not ret:
            self.failed_reads += 1
            if self.failed_reads % 30 == 0:
                self.get_logger().warn(
                    f"Camera read failed {self.failed_reads} times"
                )
            return

        # Timestamp as close as possible to successful capture
        stamp = self.get_clock().now().to_msg()

        # 1. Publish raw image
        image_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        image_msg.header.stamp = stamp
        image_msg.header.frame_id = self.frame_id
        self.image_pub.publish(image_msg)

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
            else:
                self.get_logger().warn("Failed to encode compressed image")

        # 3. Publish camera info
        if self.publish_camera_info:
            camera_info_msg = self.create_camera_info_msg(stamp)
            self.camera_info_pub.publish(camera_info_msg)

    def destroy_node(self):
        if self.cap.isOpened():
            self.cap.release()
        super().destroy_node()


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
