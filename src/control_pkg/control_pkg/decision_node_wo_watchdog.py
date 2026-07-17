import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from vision_msgs.msg import Detection2DArray
from std_msgs.msg import Bool, Float32, String
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
import math
import time


class DecisionNode(Node):
    def __init__(self):
        super().__init__('decision_node')

        self.last_target_time    = time.time()
        self.target_timeout      = 2.0
        self.obstacle_start_time = None
        self.frame_width         = 640
        self.center_x            = self.frame_width // 2
        self.turn_threshold      = 60
        self.stop_width          = 450
        self.target_id           = None
        self.obstacle_detected   = False
        self.front_distance      = 999.0
        self.free_direction      = 'LEFT'

        # ── Voice / mode state ────────────────────────────
        # voice_mode: 'AUTO'      — roam freely, avoiding obstacles (default)
        #                           and, if follow_enabled, chase a person
        #             'STOP'     — hard stop, ignore everything, indefinite
        #             'MANEUVER' — running a timed turn/forward/backward move,
        #                          then returns automatically to whatever mode
        #                          was active before it started
        self.voice_mode          = 'AUTO'
        self.follow_enabled      = False   # only True after "follow me"/"search"

        # Maneuver (timed turn / timed forward-backward) state
        self.maneuver_twist          = (0.0, 0.0)  # (linear.x, angular.z)
        self.maneuver_duration       = 0.0
        self.maneuver_start_time     = 0.0
        self.maneuver_return_mode    = 'AUTO'

        # Tunables for the timed moves
        self.turn_speed_rad_s   = 0.6            # matches your old LEFT/RIGHT speed
        self.turn_angle_rad     = math.radians(90.0)
        self.linear_speed_m_s   = 0.25
        self.linear_distance_m  = 0.20           # ~8 inches

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        self.detection_sub = self.create_subscription(
            Detection2DArray, '/tracked_detections',
            self.detection_callback, qos
        )
        self.obstacle_sub = self.create_subscription(
            Bool, '/obstacle_detected',
            self.obstacle_callback, qos
        )
        self.distance_sub = self.create_subscription(
            Float32, '/front_distance',
            self.distance_callback, qos
        )
        self.direction_sub = self.create_subscription(
            String, '/free_direction',
            self.direction_callback, qos
        )
        self.voice_sub = self.create_subscription(
            String, '/voice_command',
            self.voice_callback, qos
        )
        # When audio (STT wake-word listening) is toggled OFF, voice control
        # isn't available, so auto-switch to following. When it comes back
        # ON, hand control back to voice (revert to plain roam/AUTO).
        self.audio_sub = self.create_subscription(
            Bool, '/audio_enabled',
            self.audio_enabled_callback, qos
        )

        self.pub = self.create_publisher(Twist, '/cmd_vel', qos)

        # Drives STOP and MANEUVER states independently of the detection
        # topic's rate, so timed turns/moves are accurate regardless of
        # camera FPS.
        self.control_timer = self.create_timer(0.05, self._control_loop)  # 20 Hz

        self.get_logger().info('Decision node started — default mode: AUTO/ROAM')

    # ─────────────────────────────────────────────────────
    # Sensor callbacks — unchanged
    # ─────────────────────────────────────────────────────

    def obstacle_callback(self, msg):
        self.obstacle_detected = msg.data

    def distance_callback(self, msg):
        self.front_distance = msg.data

    def direction_callback(self, msg):
        self.free_direction = msg.data

    # ─────────────────────────────────────────────────────
    # Voice callback — maps STT commands to voice_mode
    # ─────────────────────────────────────────────────────

    def audio_enabled_callback(self, msg: Bool):
        if not msg.data:
            # Audio just went OFF — no voice control available, so follow
            # the person automatically instead of just roaming blind.
            self.voice_mode      = 'AUTO'
            self.follow_enabled  = True
            self.target_id       = None
            self.last_target_time = time.time()
            self.get_logger().info('🔇 Audio OFF — auto-switching to FOLLOW mode')
        else:
            # Audio back ON — hand control back to voice; revert to plain
            # roam until the person says "follow me" again.
            self.voice_mode      = 'AUTO'
            self.follow_enabled  = False
            self.get_logger().info('🔊 Audio ON — voice control restored, back to roam')

    def voice_callback(self, msg: String):
        cmd = msg.data.strip().upper()

        if cmd == 'WAKE_WORD_DETECTED':
            self.voice_mode = 'STOP'
            self._stop()
            self.get_logger().info('🎤 [VOICE] Wake word — motors stopped, awaiting command')
            return

        if cmd == 'RESUME':
            self.voice_mode = 'AUTO'
            self.get_logger().info('⏱ [VOICE] No command — resuming AUTO/ROAM')
            return

        if cmd == 'STOP':
            self.voice_mode     = 'STOP'
            self.follow_enabled = False
            self._stop()
            self.get_logger().info('🛑 [VOICE] STOP — halting indefinitely')
            return

        if cmd in ('FOLLOW_PERSON', 'SEARCH'):
            self.voice_mode      = 'AUTO'
            self.follow_enabled  = True
            self.target_id       = None
            self.last_target_time = time.time()
            self.get_logger().info('🤖 [VOICE] Follow mode ENABLED — searching for a person')
            return

        if cmd in ('LEFT', 'RIGHT', 'FORWARD', 'BACKWARD'):
            self._start_maneuver(cmd)
            return

        if cmd == 'DESCRIBE_SCENE':
            # No motion change — llm_node handles the actual vision call.
            self.get_logger().info('[VOICE] DESCRIBE_SCENE — no motion change')
            return

        # Unrecognised token — fall back to AUTO/ROAM (follow state preserved)
        self.get_logger().warn(f'[VOICE] Unrecognised: "{cmd}" — staying/returning to AUTO')
        if self.voice_mode not in ('AUTO',):
            self.voice_mode = 'AUTO'

    # ─────────────────────────────────────────────────────
    # Timed maneuvers: turn ~90°, move ~8" forward/backward
    # ─────────────────────────────────────────────────────

    def _start_maneuver(self, cmd):
        # Remember what to go back to once the maneuver finishes.
        self.maneuver_return_mode = self.voice_mode if self.voice_mode != 'MANEUVER' else 'AUTO'

        if cmd == 'LEFT':
            self.maneuver_twist = (0.0, self.turn_speed_rad_s)
            self.maneuver_duration = self.turn_angle_rad / self.turn_speed_rad_s
        elif cmd == 'RIGHT':
            self.maneuver_twist = (0.0, -self.turn_speed_rad_s)
            self.maneuver_duration = self.turn_angle_rad / self.turn_speed_rad_s
        elif cmd == 'FORWARD':
            self.maneuver_twist = (self.linear_speed_m_s, 0.0)
            self.maneuver_duration = self.linear_distance_m / self.linear_speed_m_s
        elif cmd == 'BACKWARD':
            self.maneuver_twist = (-self.linear_speed_m_s, 0.0)
            self.maneuver_duration = self.linear_distance_m / self.linear_speed_m_s

        self.maneuver_start_time = time.time()
        self.voice_mode = 'MANEUVER'
        self.get_logger().info(
            f'🎯 [VOICE] Maneuver {cmd} — {self.maneuver_duration:.2f}s, '
            f'will return to {self.maneuver_return_mode}'
        )

    def _control_loop(self):
        if self.voice_mode == 'MANEUVER':
            elapsed = time.time() - self.maneuver_start_time
            if elapsed >= self.maneuver_duration:
                self._stop()
                self.voice_mode = self.maneuver_return_mode
                self.get_logger().info(f'[VOICE] Maneuver complete → {self.voice_mode}')
            else:
                t = Twist()
                t.linear.x, t.angular.z = self.maneuver_twist
                self.pub.publish(t)
        elif self.voice_mode == 'STOP':
            self._stop()
        # AUTO mode motion is driven by detection_callback (needs live detections)

    def _stop(self):
        self.pub.publish(Twist())

    # ─────────────────────────────────────────────────────
    # Detection callback — roam by default, follow only if enabled
    # ─────────────────────────────────────────────────────

    def detection_callback(self, msg):
        # Only drive from detections while in AUTO. STOP/MANEUVER are
        # handled by the control loop above.
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
                if self.free_direction == 'LEFT':
                    twist.angular.z = 0.8
                else:
                    twist.angular.z = -0.8

            self.pub.publish(twist)
            self.get_logger().warn(
                f'Obstacle detected | Distance: {self.front_distance:.2f} m'
            )
            return
        else:
            self.obstacle_start_time = None

        # Only look for people at all if "follow me" / "search" was said.
        person_detections = []
        if self.follow_enabled:
            for det in msg.detections:
                if len(det.results) == 0:
                    continue
                if det.results[0].hypothesis.class_id == 'person':
                    person_detections.append(det)

        if len(person_detections) == 0:
            if self.follow_enabled:
                # We were following but lost the target — brief pause,
                # then fall back to roaming.
                elapsed = time.time() - self.last_target_time
                if elapsed < self.target_timeout:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    self.pub.publish(twist)
                    self.get_logger().info('Temporary target loss')
                    return

            # Free roam (default behaviour, and fallback when no target)
            twist.linear.x  = 0.4
            twist.angular.z = 0.15
            self.pub.publish(twist)
            self.get_logger().info('Roaming freely')
            return

        target = None
        if self.target_id is not None:
            for det in person_detections:
                if det.id == self.target_id:
                    target = det
                    break

        if target is None:
            target = max(
                person_detections,
                key=lambda d: d.bbox.size_x * d.bbox.size_y
            )
            self.target_id = target.id

        self.last_target_time = time.time()
        cx    = target.bbox.center.position.x
        width = target.bbox.size_x
        error = cx - self.center_x

        if width > self.stop_width:
            twist.linear.x  = 0.0
            twist.angular.z = 0.0
            self.get_logger().info(f'Target close stop | ID:{self.target_id}')

        elif width > 250:
            twist.linear.x  = 0.08
            twist.angular.z = -error * 0.002

        else:
            twist.linear.x  = 0.5
            twist.angular.z = -error * 0.003

        if abs(error) < 25:
            twist.angular.z = 0.0

        self.get_logger().info(
            f'Following target | Error:{error} | Distance:{self.front_distance:.2f} m'
        )
        self.pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = DecisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()