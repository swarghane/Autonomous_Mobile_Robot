import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32
from sensor_msgs.msg import Image
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import cv2
import base64
import os
import threading
import time

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(dotenv_path="/workspace/.env", override=True)


class LLMNode(Node):
    def __init__(self):
        super().__init__('llm_node')
        self._req_count = 0
        self.last_request = 0
        self.declare_parameter('min_interval', 4.0)
        self.min_interval = self.get_parameter('min_interval').value

        self.bridge = CvBridge()
        self.latest_frame = None
        self.frame_lock = threading.Lock()
        self.last_vision_time = 0
        self.declare_parameter('vision_cooldown', 3.0)
        self.vision_cooldown = self.get_parameter('vision_cooldown').value

        self.latest_battery_pct = None

        api_key = os.environ.get('NVIDIA_API_KEY', '').strip()
        if not api_key:
            self.get_logger().error('❌ NVIDIA_API_KEY not found in /workspace/.env!')
        if not api_key.startswith('nvapi-'):
            self.get_logger().error(f'❌ Key looks wrong: "{api_key[:10]}..."')

        self.client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=api_key,
            timeout=10.0
        )

        self.declare_parameter('chat_model', 'meta/llama-3.1-8b-instruct')
        self.chat_model = self.get_parameter('chat_model').value
        self.declare_parameter('vision_model', 'meta/llama-3.2-11b-vision-instruct')
        self.vision_model = self.get_parameter('vision_model').value
        self.declare_parameter('vision_max_dimension', 640)
        self.vision_max_dimension = self.get_parameter('vision_max_dimension').value
        self.declare_parameter('vision_jpeg_quality', 80)
        self.vision_jpeg_quality = self.get_parameter('vision_jpeg_quality').value

        self.declare_parameter('llm_response_topic', '/llm_response')
        self.declare_parameter('robot_ui_mode_topic', '/robot_ui_mode')
        self.declare_parameter('voice_command_topic', '/voice_command')
        self.declare_parameter('camera_topic', '/camera/image_raw')
        self.declare_parameter('battery_topic', '/battery_status')
        self.llm_response_topic = self.get_parameter('llm_response_topic').value
        self.robot_ui_mode_topic = self.get_parameter('robot_ui_mode_topic').value
        self.voice_command_topic = self.get_parameter('voice_command_topic').value
        self.camera_topic = self.get_parameter('camera_topic').value
        self.battery_topic = self.get_parameter('battery_topic').value
        self.pub = self.create_publisher(String, self.llm_response_topic, 10)
        self.ui_pub = self.create_publisher(String, self.robot_ui_mode_topic, 10)

        self._last_cmd = None
        self._last_cmd_time = 0.0
        self.declare_parameter('dedupe_window', 3.0)
        self._dedupe_window = self.get_parameter('dedupe_window').value

        self.create_subscription(String, self.voice_command_topic, self.command_callback, 10)
        self.create_subscription(
            Image, self.camera_topic, self.camera_callback,
            qos_profile_sensor_data
        )
        self.create_subscription(Float32, self.battery_topic, self._battery_callback, 10)

        self.get_logger().info(f'★ LLM ready — chat: {self.chat_model}')

    def command_callback(self, msg: String):
        t_received = time.time()
        cmd = msg.data.strip()
        cmd_lower = cmd.lower()
        cmd_upper = cmd.upper()

        now = time.time()
        if cmd_upper == self._last_cmd and (now - self._last_cmd_time) < self._dedupe_window:
            self.get_logger().info(f'[LLM] Duplicate command ignored: "{cmd}"')
            return
        self._last_cmd = cmd_upper
        self._last_cmd_time = now

        if 'face' in cmd_lower:
            self._publish_ui('face')
            self._publish_response("Okay, showing my face.")
            return

        if 'vision' in cmd_lower or 'feed' in cmd_lower or 'camera' in cmd_lower:
            self._publish_ui('vision')
            self._publish_response("Okay, showing the camera feed.")
            return

        if cmd_upper == 'BATTERY_STATUS' or any(x in cmd_lower for x in [
            'battery', 'batteries', 'batery', 'battery level', 'battery status',
            'battery percentage', 'power level', 'power status', 'charge level',
            'remaining charge', 'remaining battery',
        ]):
            self._report_battery()
            return

        if cmd_upper == 'DESCRIBE_SCENE' or 'describe' in cmd_lower:
            self._trigger_vision(t_received)
            return

        if ('explore' in cmd_lower or 'roam' in cmd_lower) and 'stop' not in cmd_lower:
            self._publish_response("Back to roaming and exploring on my own.")
            return

        control_cmds = {
            'WAKE_WORD_DETECTED', 'RESUME', 'STOP', 'FORWARD', 'BACKWARD',
            'LEFT', 'RIGHT', 'NONE', 'FOLLOW_PERSON', 'SEARCH', 'DESCRIBE_SCENE'
        }
        if cmd_upper in control_cmds:
            return

        now = time.time()
        if now - self.last_request < self.min_interval:
            self.get_logger().warn('[LLM] Too soon — request dropped.')
            return
        self.last_request = now

        threading.Thread(target=self._query_chat, args=(cmd, t_received), daemon=True).start()

    def camera_callback(self, msg: Image):
        with self.frame_lock:
            first_frame = self.latest_frame is None
            self.latest_frame = msg
        if first_frame:
            self.get_logger().info('[LLM] ✅ First camera frame received.')

    def _trigger_vision(self, t_received=None):
        now = time.time()
        if now - self.last_vision_time < self.vision_cooldown:
            self.get_logger().warn('[LLM] Vision request too soon — dropped.')
            return

        with self.frame_lock:
            frame = self.latest_frame

        if frame is None:
            self._publish_response("I don't have a camera frame yet, try again in a moment.")
            return

        self.last_vision_time = now
        threading.Thread(target=self._query_vision, args=(frame, t_received), daemon=True).start()

    def _battery_callback(self, msg: Float32):
        self.latest_battery_pct = msg.data

    def _report_battery(self):
        if self.latest_battery_pct is None:
            self._publish_response("I don't have a battery reading yet.")
            return
        pct = self.latest_battery_pct
        self._publish_response(f"My battery is at {pct:.0f} percent.")

    def _publish_ui(self, mode: str):
        out = String()
        out.data = mode
        self.ui_pub.publish(out)
        self.get_logger().info(f'[UI] /robot_ui_mode → {mode}')

    def _query_chat(self, cmd: str, t_received=None):
        self._req_count += 1
        t0 = time.time()
        self.get_logger().info(f'[LLM] Query #{self._req_count}: "{cmd}"')
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are Vector, an autonomous mobile robot who helps as an assistant robot. "
                            "Give a friendly, clear, plain text answer in 10 to 15 words. "
                            "Do not use actions, emojis, asterisks, or formatting."
                        )
                    },
                    {"role": "user", "content": cmd}
                ],
                max_tokens=60,
                temperature=0.7
            )
            t1 = time.time()
            self.get_logger().info(f'[TIMING][chat] API call: {t1 - t0:.2f}s')

            text = response.choices[0].message.content
            if not text:
                self.get_logger().warn(f'[LLM] Empty response — finish_reason: {response.choices[0].finish_reason}')
                self._publish_response("Sorry, I did not get a response.")
                return

            text = text.strip().replace('*', '').replace('_', '').replace('#', '')
            self._publish_response(text)
            t2 = time.time()
            self.get_logger().info(f'[TIMING][chat] TOTAL query_start → published: {t2 - t0:.2f}s')
            if t_received is not None:
                self.get_logger().info(
                    f'[TIMING][chat] TOTAL command_received → published: {t2 - t_received:.2f}s'
                )

        except Exception as e:
            self.get_logger().error(f'[LLM] Chat failed: {type(e).__name__}: {e}')

    def _query_vision(self, img_msg: Image, t_received=None):
        try:
            t0 = time.time()
            cv_frame = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            t1 = time.time()

            h, w = cv_frame.shape[:2]
            longer_side = max(h, w)
            if longer_side > self.vision_max_dimension:
                scale = self.vision_max_dimension / longer_side
                cv_frame = cv2.resize(
                    cv_frame,
                    (int(w * scale), int(h * scale)),
                    interpolation=cv2.INTER_AREA
                )
            t2 = time.time()

            _, buffer = cv2.imencode(
                ".jpg", cv_frame, [int(cv2.IMWRITE_JPEG_QUALITY), self.vision_jpeg_quality]
            )
            base64_image = base64.b64encode(buffer).decode("utf-8")
            t3 = time.time()
            self.get_logger().info(
                f'[TIMING][vision] cv_convert={t1-t0:.2f}s resize={t2-t1:.2f}s '
                f'encode={t3-t2:.2f}s frame={cv_frame.shape[1]}x{cv_frame.shape[0]} '
                f'payload_kb={len(base64_image)/1024:.0f}'
            )

            response = self.client.chat.completions.create(
                model=self.vision_model,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "You are Vector, an autonomous mobile robot. "
                                "In one or two sentences under 25 words, describe what you see. "
                                "Note any people, paths, or obstacles. Plain text only."
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}
                        }
                    ]
                }],
                max_tokens=50,
                temperature=0.4
            )
            t4 = time.time()
            self.get_logger().info(f'[TIMING][vision] API call: {t4-t3:.2f}s | TOTAL: {t4-t0:.2f}s')

            text = response.choices[0].message.content
            if not text:
                self._publish_response("I couldn't make out a clear description.")
                return
            text = text.strip().replace('*', '').replace('_', '').replace('#', '')
            self._publish_response(text)
            if t_received is not None:
                self.get_logger().info(
                    f'[TIMING][vision] TOTAL command_received → published: {time.time() - t_received:.2f}s'
                )

        except Exception as e:
            self.get_logger().error(f'[LLM] Vision failed: {type(e).__name__}: {e}')
            self._publish_response("Sorry, my vision check failed.")

    def _publish_response(self, text: str):
        self.get_logger().info(f'[LLM] → "{text}"')
        out = String()
        out.data = text.strip()
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = LLMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if rclpy.ok():
            node.get_logger().info('LLM node stopping...')
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()