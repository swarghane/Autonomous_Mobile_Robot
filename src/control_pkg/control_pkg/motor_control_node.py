import threading
import time

import rclpy
import serial
from geometry_msgs.msg import Twist
from rclpy.node import Node
from std_msgs.msg import Float32


class MotorControlNode(Node):
    """Convert velocity commands to motor PWM and publish battery percentage."""

    def __init__(self):
        super().__init__("motor_control_node")

        # ROS interfaces and serial communication.
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("battery_status_topic", "/battery_status")
        self.declare_parameter("serial_port", "/dev/ttyACM0")
        self.declare_parameter("baudrate", 115200)
        self.declare_parameter("serial_timeout", 1.0)

        # Velocity-to-PWM conversion.
        self.declare_parameter("stop_linear_threshold", 0.02)
        self.declare_parameter("stop_angular_threshold", 0.02)
        self.declare_parameter("search_turn_multiplier", 250.0)
        self.declare_parameter("base_speed_multiplier", 250.0)
        self.declare_parameter("follow_turn_multiplier", 150.0)
        self.declare_parameter("right_motor_correction", 0.90)
        self.declare_parameter("max_pwm", 255)
        self.declare_parameter("motor_deadzone", 15)

        # Serial recovery timing.
        self.declare_parameter("reconnect_delay", 2.0)
        self.declare_parameter("serial_error_delay", 1.0)

        # Battery ADC and voltage-divider calibration.
        self.declare_parameter("adc_max_raw", 4095)
        self.declare_parameter("adc_vref", 3.3)
        self.declare_parameter("r1_ohms", 100000.0)
        self.declare_parameter("r2_ohms", 33000.0)
        self.declare_parameter("battery_full_voltage", 12.6)
        self.declare_parameter("battery_empty_voltage", 9.0)

        self.cmd_vel_topic = str(self.get_parameter("cmd_vel_topic").value)
        self.battery_status_topic = str(
            self.get_parameter("battery_status_topic").value
        )
        self.serial_port = str(self.get_parameter("serial_port").value)
        self.baudrate = int(self.get_parameter("baudrate").value)
        self.serial_timeout = float(self.get_parameter("serial_timeout").value)

        self.stop_linear_threshold = float(
            self.get_parameter("stop_linear_threshold").value
        )
        self.stop_angular_threshold = float(
            self.get_parameter("stop_angular_threshold").value
        )
        self.search_turn_multiplier = float(
            self.get_parameter("search_turn_multiplier").value
        )
        self.base_speed_multiplier = float(
            self.get_parameter("base_speed_multiplier").value
        )
        self.follow_turn_multiplier = float(
            self.get_parameter("follow_turn_multiplier").value
        )
        self.right_motor_correction = float(
            self.get_parameter("right_motor_correction").value
        )
        self.max_pwm = int(self.get_parameter("max_pwm").value)
        self.motor_deadzone = int(self.get_parameter("motor_deadzone").value)

        self.reconnect_delay = float(
            self.get_parameter("reconnect_delay").value
        )
        self.serial_error_delay = float(
            self.get_parameter("serial_error_delay").value
        )

        self.adc_max_raw = int(self.get_parameter("adc_max_raw").value)
        self.adc_vref = float(self.get_parameter("adc_vref").value)
        self.r1_ohms = float(self.get_parameter("r1_ohms").value)
        self.r2_ohms = float(self.get_parameter("r2_ohms").value)
        self.divider_ratio = self.r2_ohms / (self.r1_ohms + self.r2_ohms)
        self.battery_full_voltage = float(
            self.get_parameter("battery_full_voltage").value
        )
        self.battery_empty_voltage = float(
            self.get_parameter("battery_empty_voltage").value
        )

        self.cmd_vel_sub = self.create_subscription(
            Twist,
            self.cmd_vel_topic,
            self.cmd_callback,
            10,
        )
        self.battery_pub = self.create_publisher(
            Float32,
            self.battery_status_topic,
            10,
        )

        self.ser = None
        try:
            self.ser = serial.Serial(
                self.serial_port,
                self.baudrate,
                timeout=self.serial_timeout,
            )
            self.get_logger().info(
                f"ESP32 serial connected on {self.serial_port}"
            )
        except Exception as error:
            self.get_logger().error(f"Serial connection failed: {error}")

        # Read battery messages without blocking motor-command writes.
        self.stop_flag = False
        self.reader_thread = threading.Thread(
            target=self._serial_read_loop,
            daemon=True,
        )
        self.reader_thread.start()

        self.get_logger().info("Motor control node started")

    def cmd_callback(self, msg: Twist):
        """Convert a Twist command into left and right motor PWM values."""
        linear = msg.linear.x
        angular = msg.angular.z

        if (
            abs(linear) < self.stop_linear_threshold
            and abs(angular) < self.stop_angular_threshold
        ):
            left_speed = 0
            right_speed = 0
        elif (
            abs(linear) < self.stop_linear_threshold
            and abs(angular) >= self.stop_angular_threshold
        ):
            turn_speed = int(angular * self.search_turn_multiplier)
            left_speed = -turn_speed
            right_speed = turn_speed
        else:
            base_speed = int(linear * self.base_speed_multiplier)
            turn_speed = int(angular * self.follow_turn_multiplier)
            left_speed = base_speed - turn_speed
            right_speed = base_speed + turn_speed
            right_speed = int(right_speed * self.right_motor_correction)

        left_speed = max(min(left_speed, self.max_pwm), -self.max_pwm)
        right_speed = max(min(right_speed, self.max_pwm), -self.max_pwm)

        if abs(left_speed) < self.motor_deadzone:
            left_speed = 0
        if abs(right_speed) < self.motor_deadzone:
            right_speed = 0

        command = f"{left_speed},{right_speed}\n"
        try:
            if self.ser:
                self.ser.write(command.encode())
        except Exception as error:
            self.get_logger().error(f"Serial write failed: {error}")

    def _serial_read_loop(self):
        """Read battery ADC messages from the ESP32 in a background thread."""
        while not self.stop_flag:
            if self.ser is None:
                time.sleep(self.reconnect_delay)
                self._reconnect_serial()
                continue

            try:
                line = (
                    self.ser.readline()
                    .decode("utf-8", errors="ignore")
                    .strip()
                )
                if not line.startswith("BATT:"):
                    continue

                raw = int(line.split(":", 1)[1])
                percent = self._raw_to_percent(raw)

                battery_msg = Float32()
                battery_msg.data = percent
                self.battery_pub.publish(battery_msg)
                self.get_logger().info(f"[BATTERY] {percent:.1f}%")
            except Exception as error:
                self.get_logger().warn(
                    f"[BATTERY] Serial read error: {error}"
                )
                time.sleep(self.serial_error_delay)

    def _reconnect_serial(self):
        """Attempt to reopen the configured serial port."""
        try:
            self.ser = serial.Serial(
                self.serial_port,
                self.baudrate,
                timeout=self.serial_timeout,
            )
            self.get_logger().info(
                f"Reconnected to {self.serial_port}"
            )
        except Exception:
            self.ser = None

    def _raw_to_percent(self, raw: int) -> float:
        """Convert an ADC sample into a clamped battery percentage."""
        raw = max(0, min(self.adc_max_raw, raw))
        adc_voltage = (raw / self.adc_max_raw) * self.adc_vref
        battery_voltage = adc_voltage / self.divider_ratio
        voltage_span = (
            self.battery_full_voltage - self.battery_empty_voltage
        )
        percent = (
            (battery_voltage - self.battery_empty_voltage)
            / voltage_span
            * 100.0
        )
        return max(0.0, min(100.0, percent))


def main(args=None):
    rclpy.init(args=args)
    node = MotorControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if rclpy.ok():
            node.get_logger().info("Motor control node stopping...")
    finally:
        node.stop_flag = True

        # Send a final motor stop before closing the serial connection.
        if node.ser:
            try:
                node.ser.write(b"0,0\n")
                node.ser.flush()
                node.ser.close()
                node.get_logger().info("Serial port closed cleanly")
            except Exception as error:
                node.get_logger().error(
                    "Failed to send stop command during shutdown: "
                    f"{error}"
                )

        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
