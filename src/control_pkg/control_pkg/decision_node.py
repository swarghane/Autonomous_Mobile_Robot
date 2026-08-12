import math
import time

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import Bool, Float32, String
from vision_msgs.msg import Detection2DArray


class DecisionNode(Node):
    """Coordinate autonomous roaming, person following, voice control, and safety stops."""

    def __init__(self):
        super().__init__('decision_node')

        self.last_target_time = time.time()
        self.target_timeout = 2.0
        self.obstacle_start_time = None
        self.frame_width = 640
        self.center_x = self.frame_width // 2
        self.turn_threshold = 60
        self.stop_width = 450
        self.target_id = None
        self.obstacle_detected = False
        self.front_distance = 999.0
        self.free_direction = 'LEFT'

        # AUTO roams or follows, STOP blocks motion, and MANEUVER runs a timed command.
        self.voice_mode = 'AUTO'
        self.follow_enabled = False
        self.pre_wake_voice_mode = self.voice_mode
        self.pre_wake_follow_enabled = self.follow_enabled

        self.maneuver_twist = (0.0, 0.0)
        self.maneuver_duration = 0.0
        self.maneuver_start_time = 0.0
        self.maneuver_return_mode = 'AUTO'

        # The measured turn rate compensates for real-world motor response.
        self.turn_speed_rad_s = 0.6
        self.measured_turn_rate_rad_s = 1.2
        self.turn_angle_rad = math.radians(90.0)
        self.linear_speed_m_s = 0.25
        self.linear_distance_m = 0.20

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray,
            '/tracked_detections',
            self.detection_callback,
            qos,
        )
        self.obstacle_sub = self.create_subscription(
            Bool,
            '/obstacle_detected',
            self.obstacle_callback,
            qos,
        )
        self.distance_sub = self.create_subscription(
            Float32,
            '/front_distance',
            self.distance_callback,
            qos,
        )
        self.direction_sub = self.create_subscription(
            String,
            '/free_direction',
            self.direction_callback,
            qos,
        )
        self.voice_sub = self.create_subscription(
            String,
            '/voice_command',
            self.voice_callback,
            qos,
        )
        self.audio_sub = self.create_subscription(
            Bool,
            '/audio_enabled',
            self.audio_enabled_callback,
            qos,
        )

        # The watchdog emergency stop overrides every motion mode.
        self.estop_active = False
        self.estop_sub = self.create_subscription(
            Bool,
            '/emergency_stop',
            self.emergency_stop_callback,
            qos,
        )

        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', qos)

        # STOP and MANEUVER remain active independently of camera callback frequency.
        self.control_timer = self.create_timer(0.05, self._control_loop)

        self.get_logger().info('Decision node started — default mode: AUTO/ROAM')

    def obstacle_callback(self, msg: Bool):
        self.obstacle_detected = msg.data

    def distance_callback(self, msg: Float32):
        self.front_distance = msg.data

    def direction_callback(self, msg: String):
        self.free_direction = msg.data

    def emergency_stop_callback(self, msg: Bool):
        self.estop_active = msg.data

        if self.estop_active:
            self._stop()
            self.get_logger().error(
                '🛑 [WATCHDOG] Emergency stop ACTIVE — overriding all motion'
            )
        else:
            self.get_logger().info(
                '✅ [WATCHDOG] Emergency stop CLEARED — resuming normal operation'
            )

    def audio_enabled_callback(self, msg: Bool):
        """Switch to following while audio is muted and restore the prior mode later."""
        if not msg.data:
            self.pre_audio_off_follow_enabled = self.follow_enabled
            self.pre_audio_off_voice_mode = (
                self.voice_mode if self.voice_mode != 'MANEUVER' else 'AUTO'
            )

            self.voice_mode = 'AUTO'
            self.follow_enabled = True
            self.target_id = None
            self.last_target_time = time.time()
            self.get_logger().info('🔇 Audio OFF — auto-switching to FOLLOW mode')
        else:
            self.voice_mode = getattr(self, 'pre_audio_off_voice_mode', 'AUTO')
            self.follow_enabled = getattr(
                self,
                'pre_audio_off_follow_enabled',
                False,
            )

            if self.follow_enabled:
                self.target_id = None
                self.last_target_time = time.time()

            self.get_logger().info(
                '🔊 Audio ON — voice control restored, resuming previous mode '
                f'(follow_enabled={self.follow_enabled})'
            )

    def voice_callback(self, msg: String):
        """Map normalized STT command tokens to motion modes and actions."""
        command = msg.data.strip().upper()

        if command == 'WAKE_WORD_DETECTED':
            self.pre_wake_voice_mode = (
                self.voice_mode if self.voice_mode != 'MANEUVER' else self.maneuver_return_mode
            )
            self.pre_wake_follow_enabled = self.follow_enabled
            self.voice_mode = 'STOP'
            self._stop()
            self.get_logger().info(
                '🎤 [VOICE] Wake word — motors stopped, awaiting command'
            )
            return

        if command == 'RESUME':
            self._restore_pre_wake_mode()
            self.get_logger().info(
                f'⏱ [VOICE] No command — resuming {self.voice_mode}'
            )
            return

        if command == 'STOP':
            self.voice_mode = 'STOP'
            self.follow_enabled = False
            self._stop()
            self.get_logger().info('🛑 [VOICE] STOP — halting indefinitely')
            return

        if command in ('FOLLOW_PERSON', 'SEARCH'):
            self.voice_mode = 'AUTO'
            self.follow_enabled = True
            self.target_id = None
            self.last_target_time = time.time()
            self.get_logger().info(
                '🤖 [VOICE] Follow mode ENABLED — searching for a person'
            )
            return

        if command in ('LEFT', 'RIGHT', 'FORWARD', 'BACKWARD'):
            self._start_maneuver(command)
            return

        if command in ('DESCRIBE_SCENE', 'BATTERY_STATUS'):
            self._restore_pre_wake_mode()
            self.get_logger().info(
                f'[VOICE] {command} — resuming {self.voice_mode}'
            )
            return

        self._restore_pre_wake_mode()
        self.get_logger().info(
            f'[VOICE] Conversation — resuming {self.voice_mode}'
        )

    def _restore_pre_wake_mode(self):
        self.voice_mode = self.pre_wake_voice_mode
        self.follow_enabled = self.pre_wake_follow_enabled

    def _start_maneuver(self, command: str):
        """Start a timed turn or linear movement and remember the return mode."""
        self.maneuver_return_mode = (
            self.voice_mode if self.voice_mode != 'MANEUVER' else 'AUTO'
        )

        if command == 'LEFT':
            self.maneuver_twist = (0.0, self.turn_speed_rad_s)
            self.maneuver_duration = (
                self.turn_angle_rad / self.measured_turn_rate_rad_s
            )
        elif command == 'RIGHT':
            self.maneuver_twist = (0.0, -self.turn_speed_rad_s)
            self.maneuver_duration = (
                self.turn_angle_rad / self.measured_turn_rate_rad_s
            )
        elif command == 'FORWARD':
            self.maneuver_twist = (self.linear_speed_m_s, 0.0)
            self.maneuver_duration = (
                self.linear_distance_m / self.linear_speed_m_s
            )
        elif command == 'BACKWARD':
            self.maneuver_twist = (-self.linear_speed_m_s, 0.0)
            self.maneuver_duration = (
                self.linear_distance_m / self.linear_speed_m_s
            )

        self.maneuver_start_time = time.time()
        self.voice_mode = 'MANEUVER'
        self.get_logger().info(
            f'🎯 [VOICE] Maneuver {command} — {self.maneuver_duration:.2f}s, '
            f'will return to {self.maneuver_return_mode}'
        )

    def _control_loop(self):
        """Maintain emergency-stop, timed-maneuver, and stopped states at 20 Hz."""
        if self.estop_active:
            self._stop()
            return

        if self.voice_mode == 'MANEUVER':
            elapsed = time.time() - self.maneuver_start_time

            if elapsed >= self.maneuver_duration:
                self._stop()
                self.voice_mode = self.maneuver_return_mode
                self.get_logger().info(
                    f'[VOICE] Maneuver complete → {self.voice_mode}'
                )
            else:
                twist = Twist()
                twist.linear.x, twist.angular.z = self.maneuver_twist
                self.cmd_vel_pub.publish(twist)
        elif self.voice_mode == 'STOP':
            self._stop()

    def _stop(self):
        self.cmd_vel_pub.publish(Twist())

    def detection_callback(self, msg: Detection2DArray):
        """Generate autonomous obstacle, roaming, and person-following commands."""
        if self.estop_active:
            self._stop()
            return

        # STOP and MANEUVER are controlled by the independent 20 Hz timer.
        if self.voice_mode != 'AUTO':
            return

        twist = Twist()

        if self.obstacle_detected:
            if self.obstacle_start_time is None:
                self.obstacle_start_time = time.time()

            elapsed = time.time() - self.obstacle_start_time

            if elapsed < 1.0:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.8 if self.free_direction == 'LEFT' else -0.8

            self.cmd_vel_pub.publish(twist)
            self.get_logger().warn(
                f'Obstacle detected | Distance: {self.front_distance:.2f} m'
            )
            return

        self.obstacle_start_time = None

        person_detections = []
        if self.follow_enabled:
            for detection in msg.detections:
                if not detection.results:
                    continue

                if detection.results[0].hypothesis.class_id == 'person':
                    person_detections.append(detection)

        if not person_detections:
            if self.follow_enabled:
                elapsed = time.time() - self.last_target_time

                if elapsed < self.target_timeout:
                    self.cmd_vel_pub.publish(twist)
                    self.get_logger().info('Temporary target loss')
                    return

            twist.linear.x = 0.4
            twist.angular.z = 0.15
            self.cmd_vel_pub.publish(twist)
            self.get_logger().info('Roaming freely')
            return

        target = None
        if self.target_id is not None:
            for detection in person_detections:
                if detection.id == self.target_id:
                    target = detection
                    break

        if target is None:
            target = max(
                person_detections,
                key=lambda detection: (
                    detection.bbox.size_x * detection.bbox.size_y
                ),
            )
            self.target_id = target.id

        self.last_target_time = time.time()
        center_x = target.bbox.center.position.x
        width = target.bbox.size_x
        error = center_x - self.center_x

        if width > self.stop_width:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.get_logger().info(
                f'Target close stop | ID:{self.target_id}'
            )
        elif width > 250:
            twist.linear.x = 0.08
            twist.angular.z = -error * 0.002
        else:
            twist.linear.x = 0.5
            twist.angular.z = -error * 0.003

        if abs(error) < 25:
            twist.angular.z = 0.0

        self.get_logger().info(
            f'Following target | Error:{error} | '
            f'Distance:{self.front_distance:.2f} m'
        )
        self.cmd_vel_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = DecisionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if rclpy.ok():
            node.get_logger().info('Decision node stopping...')
    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()