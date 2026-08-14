import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Empty, Float32, String
from vision_msgs.msg import Detection2DArray


class WatchdogNode(Node):
    """Monitor critical robot data and enforce an emergency stop when unsafe."""

    def __init__(self):
        super().__init__("watchdog_node")

        # ROS topics
        self.declare_parameter("scan_topic", "/scan")
        self.declare_parameter("obstacle_topic", "/obstacle_detected")
        self.declare_parameter(
            "tracked_detections_topic",
            "/tracked_detections",
        )
        self.declare_parameter("camera_topic", "/camera/heartbeat")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("battery_topic", "/battery_status")
        self.declare_parameter(
            "emergency_stop_topic",
            "/emergency_stop",
        )
        self.declare_parameter("speech_topic", "/llm_response")
        self.declare_parameter("emotion_topic", "/robot_emotion")

        # Topic-liveness thresholds
        self.declare_parameter("scan_timeout", 1.5)
        self.declare_parameter("obstacle_timeout", 1.5)
        self.declare_parameter("tracked_detection_timeout", 2.5)
        self.declare_parameter("camera_timeout", 2.0)
        self.declare_parameter("cmd_vel_timeout", 2.0)

        # Battery-warning thresholds
        self.declare_parameter("battery_warn_pct", 20.0)
        self.declare_parameter("battery_critical_pct", 10.0)
        self.declare_parameter("battery_stale_sec", 15.0)

        # Immediate LiDAR safety thresholds
        self.declare_parameter("critical_distance_m", 0.28)
        self.declare_parameter("forward_cone_deg", 25.0)
        self.declare_parameter("min_forward_linear_x", 0.05)

        self.declare_parameter("check_rate_hz", 5.0)
        self.declare_parameter("startup_grace_sec", 10.0)

        self.scan_topic = str(self.get_parameter("scan_topic").value)
        self.obstacle_topic = str(
            self.get_parameter("obstacle_topic").value
        )
        self.tracked_detections_topic = str(
            self.get_parameter("tracked_detections_topic").value
        )
        self.camera_topic = str(
            self.get_parameter("camera_topic").value
        )
        self.cmd_vel_topic = str(
            self.get_parameter("cmd_vel_topic").value
        )
        self.battery_topic = str(
            self.get_parameter("battery_topic").value
        )
        self.emergency_stop_topic = str(
            self.get_parameter("emergency_stop_topic").value
        )
        self.speech_topic = str(
            self.get_parameter("speech_topic").value
        )
        self.emotion_topic = str(
            self.get_parameter("emotion_topic").value
        )

        self.timeouts = {
            "scan": float(self.get_parameter("scan_timeout").value),
            "obstacle_detected": float(
                self.get_parameter("obstacle_timeout").value
            ),
            "tracked_detections": float(
                self.get_parameter("tracked_detection_timeout").value
            ),
            "camera": float(
                self.get_parameter("camera_timeout").value
            ),
            "cmd_vel": float(
                self.get_parameter("cmd_vel_timeout").value
            ),
        }

        self.battery_warn_pct = float(
            self.get_parameter("battery_warn_pct").value
        )
        self.battery_critical_pct = float(
            self.get_parameter("battery_critical_pct").value
        )
        self.battery_stale_sec = float(
            self.get_parameter("battery_stale_sec").value
        )

        self.critical_distance_m = float(
            self.get_parameter("critical_distance_m").value
        )
        self.forward_cone_deg = float(
            self.get_parameter("forward_cone_deg").value
        )
        self.min_forward_linear_x = float(
            self.get_parameter("min_forward_linear_x").value
        )

        self.check_rate_hz = float(
            self.get_parameter("check_rate_hz").value
        )
        self.startup_grace_sec = float(
            self.get_parameter("startup_grace_sec").value
        )

        reliable_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )
        scan_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        now = time.time()
        self.last_seen = {
            "scan": now,
            "obstacle_detected": now,
            "tracked_detections": now,
            "camera": now,
            "cmd_vel": now,
        }
        self.battery_pct = None
        self.battery_last_seen = now
        self.last_cmd_linear_x = 0.0
        self.last_cmd_angular_z = 0.0

        # Multiple faults may exist at once. The e-stop clears only when all
        # active reasons have been removed.
        self.active_reasons = set()
        self.emergency_active = False
        self.spoken_warnings = set()

        self.scan_sub = self.create_subscription(
            LaserScan,
            self.scan_topic,
            self._scan_callback,
            scan_qos,
        )
        self.obstacle_sub = self.create_subscription(
            Bool,
            self.obstacle_topic,
            self._make_liveness_callback("obstacle_detected"),
            reliable_qos,
        )
        self.detections_sub = self.create_subscription(
            Detection2DArray,
            self.tracked_detections_topic,
            self._make_liveness_callback("tracked_detections"),
            reliable_qos,
        )
        self.camera_sub = self.create_subscription(
            Empty,
            self.camera_topic,
            self._make_liveness_callback("camera"),
            1,
        )
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            self.cmd_vel_topic,
            self._cmd_vel_callback,
            reliable_qos,
        )
        self.battery_sub = self.create_subscription(
            Float32,
            self.battery_topic,
            self._battery_callback,
            reliable_qos,
        )

        self.estop_pub = self.create_publisher(
            Bool,
            self.emergency_stop_topic,
            reliable_qos,
        )

        # Directly publishing zero velocity remains effective even when the
        # decision node is the component that stopped responding.
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            self.cmd_vel_topic,
            reliable_qos,
        )
        self.speech_pub = self.create_publisher(
            String,
            self.speech_topic,
            reliable_qos,
        )
        self.emotion_pub = self.create_publisher(
            String,
            self.emotion_topic,
            reliable_qos,
        )

        self.startup_time = time.time()
        self.check_timer = self.create_timer(
            1.0 / self.check_rate_hz,
            self._check_loop,
        )

        self.get_logger().info(
            "Watchdog started - monitoring critical topics and battery"
        )

    def _make_liveness_callback(self, key):
        """Create a callback that records when a monitored topic is received."""

        def callback(_msg):
            self.last_seen[key] = time.time()

        return callback

    def _cmd_vel_callback(self, msg: Twist):
        """Track command liveness and whether the robot is moving forward."""
        if self.emergency_active:
            return

        self.last_seen["cmd_vel"] = time.time()
        self.last_cmd_linear_x = msg.linear.x
        self.last_cmd_angular_z = msg.angular.z

    def _scan_callback(self, msg: LaserScan):
        """Check scan liveness and detect an immediate forward collision risk."""
        self.last_seen["scan"] = time.time()

        cone = math.radians(self.forward_cone_deg)
        min_range = None

        for index, distance in enumerate(msg.ranges):
            if distance <= msg.range_min or distance > msg.range_max:
                continue

            angle = msg.angle_min + index * msg.angle_increment
            angle = math.atan2(math.sin(angle), math.cos(angle))

            if abs(angle) <= cone:
                if min_range is None or distance < min_range:
                    min_range = distance

        critical_distance = (
            min_range is not None
            and min_range < self.critical_distance_m
        )

        if critical_distance:
            if "critical_distance" not in self.active_reasons:
                self.active_reasons.add("critical_distance")
                self._speak_once(
                    "critical_distance",
                    "Obstacle very close, stopping.",
                )
                self.get_logger().error(
                    "[WATCHDOG] Critical distance "
                    f"{min_range:.2f} m while moving forward"
                )
                self._update_estop_state()
        elif "critical_distance" in self.active_reasons:
            self.active_reasons.discard("critical_distance")
            self._clear_spoken("critical_distance")
            self._update_estop_state()

    def _battery_callback(self, msg: Float32):
        """Store the latest battery percentage and reporting time."""
        if msg.data <= 0.0 or msg.data > 100.0:
            return
        
        self.battery_pct = msg.data
        self.battery_last_seen = time.time()

    def _speak_once(self, key, text):
        """Publish a warning once until the associated condition clears."""
        if key in self.spoken_warnings:
            return

        self.spoken_warnings.add(key)
        msg = String()
        msg.data = text
        self.speech_pub.publish(msg)
        self.get_logger().warn(f"[WATCHDOG] {text}")

    def _clear_spoken(self, key):
        """Allow a resolved warning to be announced again if it returns."""
        self.spoken_warnings.discard(key)

    def _set_emotion(self, emotion):
        """Publish the emotion associated with the watchdog state."""
        msg = String()
        msg.data = emotion
        self.emotion_pub.publish(msg)

    def _check_loop(self):
        """Check topic liveness, maintain the e-stop, and warn on battery."""
        now = time.time()
        startup_deadline = (
            self.startup_time + self.startup_grace_sec
        )

        if now < startup_deadline:
            return

        for key, timeout in self.timeouts.items():
            effective_last_seen = max(
                self.last_seen[key],
                startup_deadline,
            )
            age = now - effective_last_seen
            reason = f"stale_{key}"

            if key == "cmd_vel" and self.emergency_active:
                self.last_seen["cmd_vel"] = now
                if reason in self.active_reasons:
                    self.active_reasons.discard(reason)
                    self._clear_spoken(reason)
                continue

            if key == "cmd_vel":
                robot_was_moving = (
                    abs(self.last_cmd_linear_x) > 0.01
                    or abs(self.last_cmd_angular_z) > 0.01
                )

                if not robot_was_moving:
                    if reason in self.active_reasons:
                        self.active_reasons.discard(reason)
                        self._clear_spoken(reason)
                    continue

            if age > timeout:
                if reason not in self.active_reasons:
                    self.active_reasons.add(reason)
                    self._speak_once(
                        reason,
                        f'Warning, lost {key.replace("_", " ")} data.',
                    )
            elif reason in self.active_reasons:
                self.active_reasons.discard(reason)
                self._clear_spoken(reason)

        self._update_estop_state()

        if self.emergency_active:
            self._publish_failsafe_stop()

        # Startup time is also excluded from the battery-staleness age.
        effective_battery_last_seen = max(
            self.battery_last_seen,
            startup_deadline,
        )
        battery_age = now - effective_battery_last_seen

        if battery_age > self.battery_stale_sec:
            self._speak_once(
                "battery_stale",
                "Warning, battery status is not reporting.",
            )
            return

        self._clear_spoken("battery_stale")

        if self.battery_pct is None:
            return

        if self.battery_pct <= self.battery_critical_pct:
            self._speak_once(
                "battery_critical",
                "Battery critically low, Please charge me soon.",
            )
        elif self.battery_pct <= self.battery_warn_pct:
            self._clear_spoken("battery_critical")
            self._speak_once(
                "battery_warn",
                f"Battery low, {self.battery_pct:.0f} percent.",
            )
        else:
            self._clear_spoken("battery_critical")
            self._clear_spoken("battery_warn")

    def _update_estop_state(self):
        """Publish emergency-stop transitions when fault state changes."""
        should_estop = bool(self.active_reasons)

        if should_estop and not self.emergency_active:
            self.emergency_active = True
            self._publish_estop(True)
            self._publish_failsafe_stop()
            self._set_emotion("sad")
            self.get_logger().error(
                "[WATCHDOG] EMERGENCY STOP - reasons: "
                f"{self.active_reasons}"
            )
        elif not should_estop and self.emergency_active:
            self.emergency_active = False
            self._publish_estop(False)
            self._set_emotion("neutral")
            self.get_logger().info(
                "[WATCHDOG] All clear - emergency stop released"
            )

    def _publish_estop(self, active: bool):
        """Publish the current emergency-stop state."""
        msg = Bool()
        msg.data = active
        self.estop_pub.publish(msg)

    def _publish_failsafe_stop(self):
        """Publish a zero-velocity command immediately."""
        self.cmd_vel_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = WatchdogNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if rclpy.ok():
            node.get_logger().info("Watchdog node stopping...")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
