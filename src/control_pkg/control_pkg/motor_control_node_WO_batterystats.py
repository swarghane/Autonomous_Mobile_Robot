import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import serial

class MotorControlNode(Node):
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

        self.serial_port = '/dev/ttyACM0'
        self.baudrate = 115200

        try:
            self.ser = serial.Serial(
                self.serial_port,
                self.baudrate,
                timeout=1
            )
            self.get_logger().info(f'ESP32 serial connected on {self.serial_port}')
        except Exception as e:
            self.get_logger().error(f'Serial connection failed: {e}')

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
            if hasattr(self, 'ser') and self.ser:
                self.ser.write(command.encode())
        except Exception as e:
            self.get_logger().error(f'Serial write failed: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = MotorControlNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
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