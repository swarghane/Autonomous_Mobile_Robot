import math
import time
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool, Float32, String
from sensor_msgs.msg import Image, LaserScan
from vision_msgs.msg import Detection2DArray
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy, qos_profile_sensor_data


class WatchdogNode(Node):
    """
    Safety watchdog: monitors liveness of critical topics (via raw /scan
    instead of a derived /front_distance, for genuine sensor redundancy),
    reacts immediately if something is critically close ahead while
    moving forward, and monitors battery level. Publishes
    /emergency_stop=True and force-publishes zero /cmd_vel directly
    (a failsafe independent of decision_node's cooperation) whenever any
    condition is active; clears automatically once resolved. Speaks
    warnings once via the existing TTS pipeline (/llm_response), not
    repeatedly.
    """

    # ── Timeout thresholds (seconds since last message) ───────
    TIMEOUTS = {
        'scan':                1.5,   # raw lidar — independent of any node
                                       # that derives /front_distance from it
        'obstacle_detected':   1.5,
        'tracked_detections':  2.5,
        'camera':              2.0,
        'cmd_vel':             2.0,   # catches decision_node itself dying, not just sensors
    }

    # (every topic in TIMEOUTS is treated as critical-for-stop now that
    # the reason-set model drives /emergency_stop directly)

    BATTERY_WARN_PCT     = 20.0
    BATTERY_CRITICAL_PCT = 10.0
    BATTERY_STALE_SEC    = 15.0   # no /battery_status at all -> likely serial link down

    # ── Proximity-based immediate stop (reacts on every scan, not just
    # the 5Hz staleness loop — lowest possible latency for a fast obstacle) ──
    CRITICAL_DISTANCE_M   = 0.28    # stop if anything this close ahead
    FORWARD_CONE_DEG      = 25.0    # +/- degrees around straight-ahead to check
    MIN_FORWARD_LINEAR_X  = 0.05    # only counts as "moving forward" above this

    CHECK_RATE_HZ = 5.0
    STARTUP_GRACE_SEC = 10.0   # ignore staleness checks briefly while other
                               # nodes (e.g. tracker_node) are still coming up

    def __init__(self):
        super().__init__('watchdog_node')

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE
        )

        now = time.time()
        self.last_seen = {
            'scan':                now,
            'obstacle_detected':   now,
            'tracked_detections':  now,
            'camera':              now,
            'cmd_vel':             now,
        }

        self.battery_pct = None
        self.battery_last_seen = now

        self.last_cmd_linear_x = 0.0
        self.active_reasons = set()   # e.g. {'stale_scan', 'critical_distance'}
        self.emergency_active = False
        self.spoken_warnings = set()   # keys already announced, to avoid spam

        scan_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE
        )

        self.create_subscription(LaserScan, '/scan', self._scan_callback, scan_qos)
        self.create_subscription(Bool, '/obstacle_detected', self._mk_cb('obstacle_detected'), qos)
        self.create_subscription(Detection2DArray, '/tracked_detections', self._mk_cb('tracked_detections'), qos)
        self.create_subscription(Image, '/camera/image_raw', self._mk_cb('camera'), qos_profile_sensor_data)
        self.create_subscription(Twist, '/cmd_vel', self._cmd_vel_callback, qos)
        self.create_subscription(Float32, '/battery_status', self._battery_callback, qos)

        self.estop_pub = self.create_publisher(Bool, '/emergency_stop', qos)
        # Failsafe: publish zero velocity directly, in case decision_node
        # itself is unresponsive and wouldn't otherwise act on /emergency_stop.
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', qos)
        self.speech_pub = self.create_publisher(String, '/llm_response', qos)
        self.emotion_pub = self.create_publisher(String, '/robot_emotion', qos)

        self.startup_time = time.time()
        self.timer = self.create_timer(1.0 / self.CHECK_RATE_HZ, self._check_loop)

        self.get_logger().info('★ Watchdog started — monitoring critical topics + battery')

    # ─────────────────────────────────────────────────────
    def _mk_cb(self, key):
        def cb(msg):
            self.last_seen[key] = time.time()
        return cb

    def _cmd_vel_callback(self, msg: Twist):
        # Ignore our own failsafe zero-publishes so they don't mask
        # whether decision_node is genuinely still alive.
        if self.emergency_active:
            return
        self.last_seen['cmd_vel'] = time.time()
        self.last_cmd_linear_x = msg.linear.x

    def _scan_callback(self, msg: LaserScan):
        self.last_seen['scan'] = time.time()

        cone = math.radians(self.FORWARD_CONE_DEG)
        min_range = None

        for i, r in enumerate(msg.ranges):
            if r <= msg.range_min or r > msg.range_max:
                continue
            angle = msg.angle_min + i * msg.angle_increment
            # Normalize to [-pi, pi] so "straight ahead" comparisons are
            # correct regardless of how angle_min/max are defined.
            angle = math.atan2(math.sin(angle), math.cos(angle))
            if abs(angle) <= cone:
                if min_range is None or r < min_range:
                    min_range = r

        moving_forward = self.last_cmd_linear_x > self.MIN_FORWARD_LINEAR_X

        if min_range is not None and min_range < self.CRITICAL_DISTANCE_M and moving_forward:
            if 'critical_distance' not in self.active_reasons:
                self.active_reasons.add('critical_distance')
                self._speak_once('critical_distance', 'Obstacle very close, stopping.')
                self.get_logger().error(
                    f'[WATCHDOG] 🛑 Critical distance {min_range:.2f}m while moving forward'
                )
                self._update_estop_state()
        else:
            if 'critical_distance' in self.active_reasons:
                self.active_reasons.discard('critical_distance')
                self._clear_spoken('critical_distance')
                self._update_estop_state()

    def _battery_callback(self, msg: Float32):
        self.battery_pct = msg.data
        self.battery_last_seen = time.time()

    # ─────────────────────────────────────────────────────
    def _speak_once(self, key, text):
        """Speak a warning only the first time this condition triggers;
        clear the flag once the condition resolves so it can re-announce
        if it happens again later."""
        if key in self.spoken_warnings:
            return
        self.spoken_warnings.add(key)
        msg = String()
        msg.data = text
        self.speech_pub.publish(msg)
        self.get_logger().warn(f'[WATCHDOG] {text}')

    def _clear_spoken(self, key):
        self.spoken_warnings.discard(key)

    def _set_emotion(self, emotion):
        msg = String()
        msg.data = emotion
        self.emotion_pub.publish(msg)

    # ─────────────────────────────────────────────────────
    def _check_loop(self):
        now = time.time()

        if (now - self.startup_time) < self.STARTUP_GRACE_SEC:
            return

        for key, timeout in self.TIMEOUTS.items():
            age = now - self.last_seen[key]
            reason = f'stale_{key}'
            if age > timeout:
                if reason not in self.active_reasons:
                    self.active_reasons.add(reason)
                    self._speak_once(reason, f'Warning, lost {key.replace("_", " ")} data.')
            else:
                if reason in self.active_reasons:
                    self.active_reasons.discard(reason)
                    self._clear_spoken(reason)

        self._update_estop_state()

        if self.emergency_active:
            # Keep forcing zero velocity every check cycle for as long as
            # the emergency persists — this is what actually protects
            # against decision_node being the thing that's dead.
            self._publish_failsafe_stop()

        # ── Battery checks (independent of e-stop logic) ──
        battery_age = now - self.battery_last_seen
        if battery_age > self.BATTERY_STALE_SEC:
            self._speak_once('battery_stale', 'Warning, battery status is not reporting.')
        else:
            self._clear_spoken('battery_stale')

            if self.battery_pct is not None:
                if self.battery_pct <= self.BATTERY_CRITICAL_PCT:
                    self._speak_once('battery_critical', f'Battery critically low, {self.battery_pct:.0f} percent. Please charge me soon.')
                elif self.battery_pct <= self.BATTERY_WARN_PCT:
                    self._speak_once('battery_warn', f'Battery low, {self.battery_pct:.0f} percent.')
                else:
                    self._clear_spoken('battery_critical')
                    self._clear_spoken('battery_warn')

    def _update_estop_state(self):
        should_estop = bool(self.active_reasons)

        if should_estop and not self.emergency_active:
            self.emergency_active = True
            self._publish_estop(True)
            self._publish_failsafe_stop()
            self._set_emotion('sad')
            self.get_logger().error(f'[WATCHDOG] 🛑 EMERGENCY STOP — reasons: {self.active_reasons}')
        elif not should_estop and self.emergency_active:
            self.emergency_active = False
            self._publish_estop(False)
            self._set_emotion('neutral')
            self.get_logger().info('[WATCHDOG] ✅ All clear — stop released')

    def _publish_estop(self, active: bool):
        msg = Bool()
        msg.data = active
        self.estop_pub.publish(msg)

    def _publish_failsafe_stop(self):
        self.cmd_vel_pub.publish(Twist())


def main(args=None):
    rclpy.init(args=args)
    node = WatchdogNode()
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