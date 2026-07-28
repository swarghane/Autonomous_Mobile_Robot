import os
os.environ.setdefault('XDG_RUNTIME_DIR', '/run/user/1000')
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float32, Bool

import threading
import asyncio
import edge_tts
import tempfile
import os
import time

from pydub import AudioSegment


class TTSNode(Node):

    def __init__(self):
        super().__init__('tts_node')

        self.voice = "en-US-BrianNeural"

        self.status_pub = self.create_publisher(String, '/tts_status', 10)
        self.emotion_pub = self.create_publisher(String, '/robot_emotion', 10)
        self.mouth_pub = self.create_publisher(Float32, '/mouth_level', 10)

        self.loop = asyncio.new_event_loop()
        self.speech_q = None

        self.player_process = None

        self.chunk_ms = 50

        self.bt_latency_s = 1.2

        self.cache_dir = tempfile.mkdtemp(prefix="tts_cache_")
        # Every one of these is 100% fixed, predictable text — cache them
        # all so they play instantly with zero edge_tts network/synthesis
        # delay, instead of only "Yes?" and the audio toggle lines.
        self.cached_phrases = {
            'WAKE_WORD_DETECTED': 'Yes?',
            'AUDIO_ON': 'Voice commands on.',
            'AUDIO_OFF': 'Voice commands off.',
            'FOLLOW_PERSON': 'Following you.',
            'STOP': 'Stopping.',
            'SEARCH': 'Searching.',
            'FORWARD': 'Moving forward.',
            'BACKWARD': 'Moving backward.',
            'LEFT': 'Turning left.',
            'RIGHT': 'Turning right.',
            'RESUME': 'Resuming.',
            'READY': 'Vector is ready.',
        }
        self.cache_paths = {}   # phrase text -> filepath

        self._last_voice_cmd      = None
        self._last_voice_cmd_time = 0.0
        self._voice_cmd_dedupe_s  = 2.0

        threading.Thread(
            target=self._start_async_loop,
            daemon=True
        ).start()

        self.create_subscription(
            String,
            '/llm_response',
            self.response_callback,
            10
        )

        self.create_subscription(
            String,
            '/voice_command',
            self.command_callback,
            10
        )

        self._last_audio_enabled = None
        self.create_subscription(
            Bool,
            '/audio_enabled',
            self._audio_enabled_callback,
            10
        )

        self.get_logger().info("★ Edge-TTS + ffplay + lip-sync ready")

    def response_callback(self, msg):
        if self.speech_q:
            self.loop.call_soon_threadsafe(
                self.speech_q.put_nowait,
                msg.data.strip()
            )

    def command_callback(self, msg):
        cmd_upper = msg.data.strip().upper()

        now = time.time()
        if cmd_upper == self._last_voice_cmd and (now - self._last_voice_cmd_time) < self._voice_cmd_dedupe_s:
            self.get_logger().info(f'[TTS] Duplicate command ignored: "{cmd_upper}"')
            return
        self._last_voice_cmd      = cmd_upper
        self._last_voice_cmd_time = now

        confirmations = {
            'FOLLOW_PERSON': 'Following you.',
            'STOP': 'Stopping.',
            'SEARCH': 'Searching.',
            'FORWARD': 'Moving forward.',
            'BACKWARD': 'Moving backward.',
            'LEFT': 'Turning left.',
            'RIGHT': 'Turning right.',
            'RESUME': 'Resuming.',
            'WAKE_WORD_DETECTED': 'Yes?',
        }

        text = confirmations.get(cmd_upper)

        if text and self.speech_q:
            self.loop.call_soon_threadsafe(
                self.speech_q.put_nowait,
                text
            )

    def _audio_enabled_callback(self, msg: Bool):
        enabled = msg.data
        if enabled == self._last_audio_enabled:
            return
        self._last_audio_enabled = enabled

        text = "Voice commands on." if enabled else "Voice commands off."
        if self.speech_q:
            self.loop.call_soon_threadsafe(
                self.speech_q.put_nowait,
                text
            )

    def _publish_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def _publish_emotion(self, text):
        msg = String()
        msg.data = text
        self.emotion_pub.publish(msg)

    def _publish_mouth(self, level: float):
        msg = Float32()
        msg.data = max(0.0, min(1.0, level))
        self.mouth_pub.publish(msg)

    def _start_async_loop(self):
        asyncio.set_event_loop(self.loop)
        self.speech_q = asyncio.Queue()
        self.speech_q.put_nowait("Vector is ready.")
        self.loop.run_until_complete(
            self._speech_loop()
        )

    def _compute_levels(self, filename):
        try:
            audio = AudioSegment.from_mp3(filename)
        except Exception as e:
            self.get_logger().warn(f'[TTS] Could not analyze audio for lip-sync: {e}')
            return []

        raw_levels = []
        for i in range(0, len(audio), self.chunk_ms):
            chunk = audio[i:i + self.chunk_ms]
            raw_levels.append(chunk.rms)

        max_rms = max(raw_levels) if raw_levels else 1
        if max_rms == 0:
            max_rms = 1
        return [min(1.0, rms / max_rms) for rms in raw_levels]

    async def _lipsync_stream(self, levels, start_delay: float = 0.0):
        try:
            if start_delay > 0:
                await asyncio.sleep(start_delay)
            for level in levels:
                self._publish_mouth(level)
                await asyncio.sleep(self.chunk_ms / 1000.0)
        except asyncio.CancelledError:
            pass
        finally:
            self._publish_mouth(0.0)

    async def _pregenerate_cache(self):
        async def cache_one(text):
            try:
                path = os.path.join(self.cache_dir, f"{abs(hash(text))}.mp3")
                communicate = edge_tts.Communicate(text=text, voice=self.voice)
                await communicate.save(path)
                self.cache_paths[text] = path
                self.get_logger().info(f'[TTS] Cached phrase: "{text}"')
            except Exception as e:
                self.get_logger().warn(f'[TTS] Failed to pre-cache "{text}": {e}')

        # Run all cache-generation network calls concurrently instead of
        # one at a time — with 11 phrases, sequential awaits meant an
        # 11x network-latency startup delay before "Vector is ready" could
        # even play.
        await asyncio.gather(*(cache_one(t) for t in set(self.cached_phrases.values())))

    async def _speech_loop(self):
        # Launch caching in the background rather than blocking on it —
        # if something needs to be spoken before caching finishes, it
        # just falls back to live synthesis for that one instance and
        # self-heals once the cache task completes.
        asyncio.ensure_future(self._pregenerate_cache())

        while rclpy.ok():
            text = await self.speech_q.get()
            if not text:
                continue

            filename = None
            lipsync_task = None
            is_cached = text in self.cache_paths

            try:
                self._publish_status("SPEAKING")
                self.get_logger().info(f"TTS: {text}")

                if is_cached:
                    filename = self.cache_paths[text]
                else:
                    fd, filename = tempfile.mkstemp(suffix=".mp3")
                    os.close(fd)
                    communicate = edge_tts.Communicate(
                        text=text,
                        voice=self.voice
                    )
                    await communicate.save(filename)

                levels = self._compute_levels(filename)

                self._publish_emotion("talking")

                self.player_process = await asyncio.create_subprocess_exec(
                    "ffplay",
                    "-nodisp",
                    "-autoexit",
                    "-loglevel",
                    "quiet",
                    filename,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )

                if levels:
                    lipsync_task = asyncio.ensure_future(
                        self._lipsync_stream(levels, start_delay=self.bt_latency_s)
                    )

                await self.player_process.wait()
                await asyncio.sleep(0.4)  # delay is for BT speakers

            except Exception as e:
                self.get_logger().error(str(e))

            finally:
                if lipsync_task:
                    lipsync_task.cancel()
                self._publish_mouth(0.0)
                self.player_process = None
                if filename and not is_cached and os.path.exists(filename):
                    os.remove(filename)
                self._publish_status("IDLE")
                self._publish_emotion("neutral")
                self.speech_q.task_done()

    async def stop_speaking(self):
        if self.player_process:
            self.player_process.terminate()
            try:
                await asyncio.wait_for(
                    self.player_process.wait(),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                self.player_process.kill()


def main(args=None):
    rclpy.init(args=args)
    node = TTSNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if rclpy.ok():
            node.get_logger().info('TTS node stopping...')
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == '__main__':
    main()