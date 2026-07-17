import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import requests
import threading
import time

OLLAMA_URL   = 'http://172.17.0.1:11434/api/generate'  # host from Docker
OLLAMA_MODEL = 'tinyllama'   # or 'phi3:mini' if RAM allows

class LLMNode(Node):
    def __init__(self):
        super().__init__('llm_node')
        self._req_count      = 0
        self.last_request    = 0
        self.min_interval    = 4.0  # seconds between requests

        self.pub = self.create_publisher(String, '/llm_response', 10)
        self.create_subscription(
            String, '/voice_command',
            self.command_callback, 10
        )
        self.get_logger().info(f'★ LLM ready (ollama/{OLLAMA_MODEL}).')

    def command_callback(self, msg: String):
        cmd = msg.data.strip()

        control_cmds = {
            'WAKE_WORD_DETECTED', 'RESUME', 'STOP', 'FOLLOW_PERSON',
            'SEARCH', 'FORWARD', 'BACKWARD', 'LEFT', 'RIGHT', 'NONE'
        }
        if cmd.upper() in control_cmds:
            return

        # Rate limit
        now = time.time()
        if now - self.last_request < self.min_interval:
            self.get_logger().warn('[LLM] Too soon — skipping')
            return
        self.last_request = now

        # Run in thread so ROS doesn't block
        threading.Thread(
            target=self._query,
            args=(cmd,),
            daemon=True
        ).start()

    def _query(self, cmd: str):
        self._req_count += 1
        self.get_logger().info(f'[LLM] #{self._req_count} "{cmd}"')
        try:
            response = requests.post(OLLAMA_URL, json={
                'model':  OLLAMA_MODEL,
                'prompt': (
                    f'System: You are Vector, a helpful assistant robot. '
                    f'Give a friendly, clear, plain text answer in 10 to 15 words. '
                    f'Do not use actions, emojis, asterisks, or formatting.\n'
                    f'Human: {cmd}\n'
                    f'Vector:'
                ),
                'stream': False,
                'options': {
                    'num_predict': 40,  # short = fast
                    'temperature': 0.7,
                    'stop': ['Human:', '\n\n']
                }
            }, timeout=15)

            text = response.json()['response'].strip()
            text = text.replace('*', '').replace('_', '').replace('#', '')
            self.get_logger().info(f'[LLM] Response: "{text}"')

            out      = String()
            out.data = text
            self.pub.publish(out)

        except requests.exceptions.ConnectionError:
            self.get_logger().error('[LLM] Cannot reach ollama')
        except Exception as e:
            self.get_logger().error(f'[LLM] Error: {e}')


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