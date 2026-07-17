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
        self._req_count      = 0
        self.last_request    = 0
        self.min_interval    = 4.0

        self.bridge           = CvBridge()
        self.latest_frame     = None          # cached camera frame for on-demand vision
        self.frame_lock        = threading.Lock()
        self.last_vision_time = 0
        self.vision_cooldown  = 3.0            # avoid spamming vision calls

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

        self.chat_model   = 'meta/llama-3.1-8b-instruct'
        self.vision_model = 'meta/llama-3.2-11b-vision-instruct'

        self.pub    = self.create_publisher(String, '/llm_response', 10)
        # Drives the web UI's existing mode switch (see app.js: uiModeListener
        # on /robot_ui_mode, expects 'face' or 'vision', lowercase).
        self.ui_pub = self.create_publisher(String, '/robot_ui_mode', 10)

        # De-dupe: many STT/wake-word pipelines re-publish the same
        # recognised phrase 2-3 times (partial + final results, or a
        # repeated trigger while the phrase stays "active"). Without this,
        # every branch below fires once per duplicate.
        self._last_cmd       = None
        self._last_cmd_time  = 0.0
        self._dedupe_window  = 3.0   # seconds

        self.create_subscription(String, '/voice_command', self.command_callback, 10)
        self.create_subscription(
            Image, '/camera/image_raw', self.camera_callback,
            qos_profile_sensor_data   # matches typical camera driver QoS (BEST_EFFORT)
        )
        self.create_subscription(Float32, '/battery_status', self._battery_callback, 10)

        self.get_logger().info(f'★ LLM ready — chat: {self.chat_model}')

    def command_callback(self, msg: String):
        cmd       = msg.data.strip()
        cmd_lower = cmd.lower()
        cmd_upper = cmd.upper()

        # ── Global de-dupe ──────────────────────────────────
        # Ignore an identical command repeated within the dedupe window —
        # stops "follow me", "describe scene", etc. from firing 2-3x when
        # the STT node re-publishes the same recognised phrase.
        now = time.time()
        if cmd_upper == self._last_cmd and (now - self._last_cmd_time) < self._dedupe_window:
            self.get_logger().info(f'[LLM] Duplicate command ignored: "{cmd}"')
            return
        self._last_cmd      = cmd_upper
        self._last_cmd_time = now

        # ── UI control: "face" / "vision" / "feed" ─────────
        if 'face' in cmd_lower:
            self._publish_ui('face')
            self._publish_response("Okay, showing my face.")
            return

        if 'vision' in cmd_lower or 'feed' in cmd_lower or 'camera' in cmd_lower:
            self._publish_ui('vision')
            self._publish_response("Okay, showing the camera feed.")
            return

        # ── Battery status ──────────────────────────────────
        if 'battery' in cmd_lower:
            self._report_battery()
            return

        # ── Describe scene: one-shot vision query ──────────
        if cmd_upper == 'DESCRIBE_SCENE' or 'describe' in cmd_lower:
            self._trigger_vision()
            return

        # Exploration/roam and follow-me are handled entirely by
        # decision_node now (default AUTO = roam, FOLLOW_PERSON/SEARCH =
        # follow). llm_node just acknowledges so the person gets feedback.
        if 'follow' in cmd_lower:
            self._publish_response("Looking for you now. I will follow you.")
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

        threading.Thread(target=self._query_chat, args=(cmd,), daemon=True).start()

    def camera_callback(self, msg: Image):
        # Always cache the latest frame (cheap) so a "describe scene"
        # command can use an up-to-date image without a live polling loop.
        with self.frame_lock:
            first_frame = self.latest_frame is None
            self.latest_frame = msg
        if first_frame:
            self.get_logger().info('[LLM] ✅ First camera frame received.')

    def _trigger_vision(self):
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
        threading.Thread(target=self._query_vision, args=(frame,), daemon=True).start()

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

    def _query_chat(self, cmd: str):
        self._req_count += 1
        self.get_logger().info(f'[LLM] Query #{self._req_count}: "{cmd}"')
        try:
            response = self.client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are Vector, a helpful assistant robot. "
                            "Give a friendly, clear, plain text answer in 10 to 15 words. "
                            "Do not use actions, emojis, asterisks, or formatting."
                        )
                    },
                    {"role": "user", "content": cmd}
                ],
                max_tokens=60,
                temperature=0.7
            )

            text = response.choices[0].message.content
            if not text:
                self.get_logger().warn(f'[LLM] Empty response — finish_reason: {response.choices[0].finish_reason}')
                self._publish_response("Sorry, I did not get a response.")
                return

            text = text.strip().replace('*', '').replace('_', '').replace('#', '')
            self._publish_response(text)

        except Exception as e:
            self.get_logger().error(f'[LLM] Chat failed: {type(e).__name__}: {e}')

    def _query_vision(self, img_msg: Image):
        try:
            cv_frame     = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
            _, buffer    = cv2.imencode(".jpg", cv_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            base64_image = base64.b64encode(buffer).decode("utf-8")

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
                max_tokens=60,
                temperature=0.5
            )

            text = response.choices[0].message.content
            if not text:
                self._publish_response("I couldn't make out a clear description.")
                return
            text = text.strip().replace('*', '').replace('_', '').replace('#', '')
            self._publish_response(text)

        except Exception as e:
            self.get_logger().error(f'[LLM] Vision failed: {type(e).__name__}: {e}')
            self._publish_response("Sorry, my vision check failed.")

    def _publish_response(self, text: str):
        self.get_logger().info(f'[LLM] → "{text}"')
        out      = String()
        out.data = ". " + text
        self.pub.publish(out)


def main(args=None):
    rclpy.init(args=args)
    node = LLMNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()