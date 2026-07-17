import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32
import serial
import threading
import time


class MotorControlNode(Node):

    # ── Battery config (shares this node's single serial link) ────
    ADC_MAX_RAW  = 4095
    ADC_VREF     = 3.3
    R1_OHMS = 100_000.0
    R2_OHMS = 33_000.0
    DIVIDER_RATIO = R2_OHMS / (R1_OHMS + R2_OHMS)   # Vadc = Vbatt * ratio
    BATT_FULL_V  = 12.6   # 3S Li-ion/LiPo
    BATT_EMPTY_V = 9.0

    def __init__(self):
        super().__init__('motor_control_node')
        # Using a standard depth profile. This automatically matches with both
        # RELIABLE and BEST_EFFORT publishers, fixing potential QoS mismatches.
        self.sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_callback,
            10
        )

        self.battery_pub = self.create_publisher(Float32, '/battery_status', 10)

        self.serial_port = '/dev/ttyACM0'
        self.baudrate = 115200

        self.ser = None
        try:
            self.ser = serial.Serial(
                self.serial_port,
                self.baudrate,
                timeout=1
            )
            self.get_logger().info(f'ESP32 serial connected on {self.serial_port}')
        except Exception as e:
            self.get_logger().error(f'Serial connection failed: {e}')

        # Background thread reads whatever the ESP32 sends back
        # (currently just "BATT:<raw>\n" lines) without blocking
        # cmd_callback's writes.
        self._stop_flag = False
        self._reader_thread = threading.Thread(target=self._serial_read_loop, daemon=True)
        self._reader_thread.start()

        self.get_logger().info('Motor control node started')

    def cmd_callback(self, msg):
        linear = msg.linear.x
        angular = msg.angular.z

        # ---------------------------------
        # STOPPING HANDLER (Target Close / No Target)
        # ---------------------------------
        if abs(linear) < 0.02 and abs(angular) < 0.02:
            left_speed = 0
            right_speed = 0

        # ---------------------------------
        # SEARCH MODE (rotate in place)
        # ---------------------------------
        elif abs(linear) < 0.02 and abs(angular) >= 0.02:
            # Scaled turn speed to ensure it overcomes initial motor friction
            turn_speed = int(angular * 250)
            left_speed = -turn_speed
            right_speed = turn_speed

        # ---------------------------------
        # FOLLOW MODE (move + steer dynamically)
        # ---------------------------------
        else:
            # Increased baseline multiplier slightly (from 200 to 250)
            # This ensures gentle tracking commands aren't instantly choked by the deadzone
            base_speed = int(linear * 250)
            turn_speed = int(angular * 150)
            left_speed = base_speed - turn_speed
            right_speed = base_speed + turn_speed
            # Mechanical drift correction factor
            right_speed = int(right_speed * 0.90)

        # Final Clamp to standard PWM range
        left_speed = max(min(left_speed, 255), -255)
        right_speed = max(min(right_speed, 255), -255)

        # Lowered Deadzone Filter (Reduced from 35 to 15)
        # Person-following often generates small, incremental velocities.
        # A deadzone of 35 is often too high and drops legitimate tracking commands.
        if abs(left_speed) < 15:
            left_speed = 0
        if abs(right_speed) < 15:
            right_speed = 0

        # Construct and transmit command string
        command = f'{left_speed},{right_speed}\n'
        try:
            if self.ser:
                self.ser.write(command.encode())
        except Exception as e:
            self.get_logger().error(f'Serial write failed: {e}')

    # ─────────────────────────────────────────────────────
    # Battery: read "BATT:<raw>" lines the ESP32 sends back
    # over the same serial connection used for motor commands.
    # ─────────────────────────────────────────────────────

    def _serial_read_loop(self):
        while not self._stop_flag:
            if self.ser is None:
                time.sleep(2.0)
                self._reconnect_serial()
                continue
            try:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if not line.startswith('BATT:'):
                    continue
                raw = int(line.split(':', 1)[1])
                percent = self._raw_to_percent(raw)
                msg = Float32()
                msg.data = percent
                self.battery_pub.publish(msg)
                self.get_logger().info(f'[BATTERY] {percent:.1f}%')
            except Exception as e:
                self.get_logger().warn(f'[BATTERY] Serial read error: {e}')
                time.sleep(1.0)

    def _reconnect_serial(self):
        try:
            self.ser = serial.Serial(self.serial_port, self.baudrate, timeout=1)
            self.get_logger().info(f'✅ Reconnected to {self.serial_port}')
        except Exception:
            self.ser = None

    def _raw_to_percent(self, raw: int) -> float:
        raw = max(0, min(self.ADC_MAX_RAW, raw))
        v_adc  = (raw / self.ADC_MAX_RAW) * self.ADC_VREF
        v_batt = v_adc / self.DIVIDER_RATIO
        span = self.BATT_FULL_V - self.BATT_EMPTY_V
        percent = (v_batt - self.BATT_EMPTY_V) / span * 100.0
        return max(0.0, min(100.0, percent))


def main(args=None):
    rclpy.init(args=args)
    node = MotorControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._stop_flag = True
        # Emergency stop routine on shutdown
        if hasattr(node, 'ser') and node.ser:
            try:
                node.ser.write(b'0,0\n')
                node.ser.flush()
                node.ser.close()
                node.get_logger().info('Serial port closed cleanly.')
            except Exception as e:
                node.get_logger().error(f'Failed to send stop command during shutdown: {e}')
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()