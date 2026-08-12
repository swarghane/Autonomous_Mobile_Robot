import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Bool
from faster_whisper import WhisperModel
import sounddevice as sd
import numpy as np
import threading
import queue
from scipy.signal import resample_poly


class STTNode(Node):
    def __init__(self):
        super().__init__('stt_node')
        self._stop_event = threading.Event()

        self.declare_parameter('voice_command_topic', '/voice_command')
        self.declare_parameter('tts_status_topic', '/tts_status')
        self.declare_parameter('audio_enabled_topic', '/audio_enabled')

        self.declare_parameter('model_name', 'tiny.en')
        self.declare_parameter('device', 'cuda')
        self.declare_parameter('compute_type', 'int8_float16')

        self.declare_parameter('sample_rate', 44100)
        self.declare_parameter('chunk_size', 4410)
        self.declare_parameter('whisper_rate', 16000)
        self.declare_parameter('channels', 1)
        self.declare_parameter('audio_dtype', 'float32')
        self.declare_parameter('input_device', '')

        self.declare_parameter('rms_wake', 600.0)
        self.declare_parameter('rms_command', 150.0)
        self.declare_parameter('silence_sec', 0.7)
        self.declare_parameter('command_max_sec', 4.0)

        self.declare_parameter('tts_echo_decay_sec', 0.3)
        self.declare_parameter('ack_start_timeout', 0.3)
        self.declare_parameter('ack_total_timeout', 6.0)

        self.voice_command_topic = self.get_parameter('voice_command_topic').value
        self.tts_status_topic = self.get_parameter('tts_status_topic').value
        self.audio_enabled_topic = self.get_parameter('audio_enabled_topic').value

        self.model_name = self.get_parameter('model_name').value
        self.device = self.get_parameter('device').value
        self.compute_type = self.get_parameter('compute_type').value

        self.sample_rate = self.get_parameter('sample_rate').value
        self.chunk_size = self.get_parameter('chunk_size').value
        self.whisper_rate = self.get_parameter('whisper_rate').value
        self.channels = self.get_parameter('channels').value
        self.audio_dtype = self.get_parameter('audio_dtype').value
        self.input_device = self.get_parameter('input_device').value

        self.rms_wake = self.get_parameter('rms_wake').value
        self.rms_command = self.get_parameter('rms_command').value
        self.silence_sec = self.get_parameter('silence_sec').value
        self.command_max_sec = self.get_parameter('command_max_sec').value

        self.tts_echo_decay_sec = self.get_parameter('tts_echo_decay_sec').value
        self.ack_start_timeout = self.get_parameter('ack_start_timeout').value
        self.ack_total_timeout = self.get_parameter('ack_total_timeout').value

        self.pub = self.create_publisher(String, self.voice_command_topic, 10)

        self.create_subscription(
            String,
            self.tts_status_topic,
            self._tts_status_callback,
            10
        )
        self.is_tts_speaking = False

        self.audio_enabled = True
        self.create_subscription(
            Bool,
            self.audio_enabled_topic,
            self._audio_enabled_callback,
            10
        )

        self.get_logger().info(
            f'Loading Whisper ({self.model_name}, {self.device}, {self.compute_type})...'
        )
        self.model = WhisperModel(
            self.model_name,
            device=self.device,
            compute_type=self.compute_type
        )

        self.audio_q = queue.Queue()

        stream_kwargs = {
            'samplerate': self.sample_rate,
            'channels': self.channels,
            'dtype': self.audio_dtype,
            'blocksize': self.chunk_size,
            'callback': lambda i, f, t, s: self.audio_q.put(i.copy().flatten()),
        }
        if self.input_device:
            stream_kwargs['device'] = self.input_device

        self.stream = sd.InputStream(**stream_kwargs)
        self.stream.start()

        self._stop_event = threading.Event()
        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()
        self.get_logger().info('★ Ready. Say "Vector".')

    def _tts_status_callback(self, msg: String):
        status = msg.data.upper()
        if status == 'SPEAKING':
            self.is_tts_speaking = True
            self._t_speaking_start = time.time()
            self.get_logger().info('🔇 TTS Active: muting STT.')
        elif status == 'IDLE':
            t_idle_received = time.time()
            speaking_dur = t_idle_received - getattr(self, '_t_speaking_start', t_idle_received)
            self.get_logger().info(f'[TIMING] SPEAKING→IDLE received: {speaking_dur:.2f}s')
            time.sleep(self.tts_echo_decay_sec)
            self._flush_queue()
            self.is_tts_speaking = False
            self.get_logger().info('🔊 TTS Idle: STT resumed.')

    def _audio_enabled_callback(self, msg: Bool):
        was_enabled = self.audio_enabled
        self.audio_enabled = msg.data
        if was_enabled and not self.audio_enabled:
            self.get_logger().info('🔇 Audio mode OFF — ignoring wake word until re-enabled.')
            self._flush_queue()
        elif not was_enabled and self.audio_enabled:
            self.get_logger().info('🔊 Audio mode ON — listening for "Vector" again.')
            self._flush_queue()

    def _flush_queue(self):
        while not self.audio_q.empty():
            try:
                self.audio_q.get_nowait()
            except queue.Empty:
                break

    def _wait_for_ack_tts(self, start_timeout=None, total_timeout=None):
        if start_timeout is None:
            start_timeout = self.ack_start_timeout
        if total_timeout is None:
            total_timeout = self.ack_total_timeout

        start = time.time()
        while not self.is_tts_speaking and (time.time() - start) < start_timeout:
            time.sleep(0.02)
        while self.is_tts_speaking and (time.time() - start) < total_timeout:
            time.sleep(0.02)

    def _to_whisper(self, audio: np.ndarray) -> np.ndarray:
        return resample_poly(audio, self.whisper_rate, self.sample_rate).astype(np.float32)

    def _transcribe(self, audio: np.ndarray, prompt: str) -> str:
        audio_16k = self._to_whisper(audio)

        segments, info = self.model.transcribe(
            audio_16k,
            language='en',
            initial_prompt=prompt,
            beam_size=1,
            best_of=1,
            temperature=0.0,
            no_speech_threshold=0.6,
        )

        text_parts = []
        for seg in segments:
            if getattr(seg, 'compression_ratio', 0) > 2.4:
                self.get_logger().warn('[STT] Hallucination detected — ignored')
                return ''
            text_parts.append(seg.text)

        text = ''.join(text_parts).lower().strip()

        words = text.split()
        if len(words) > 5:
            top = max(set(words), key=words.count)
            if words.count(top) / len(words) > 0.5:
                self.get_logger().warn(f'[STT] Repetition loop on "{top}" — ignored')
                return ''

        return text

    def _collect_until_silence(self, max_sec=6.0, prefill=None) -> np.ndarray:
        silence_chunks = int(self.silence_sec * self.sample_rate / self.chunk_size)
        max_chunks = int(max_sec * self.sample_rate / self.chunk_size)

        buf = list(prefill) if prefill else []
        silent = 0
        started = False

        for chunk in buf:
            rms = np.sqrt(np.mean(chunk ** 2)) * 32767
            if rms > self.rms_command:
                started = True
                break

        self.get_logger().info(
            f'[CMD] collecting — prefill:{len(buf)} chunks, '
            f'voice_detected:{started}, threshold:{self.rms_command}'
        )

        remaining = max_chunks - len(buf)
        for _ in range(remaining):
            if self.is_tts_speaking or not self.audio_enabled:
                return np.array([], dtype=np.float32)

            try:
                chunk = self.audio_q.get(timeout=0.3)
            except queue.Empty:
                if started:
                    silent += 1
                    if silent >= silence_chunks:
                        break
                continue

            buf.append(chunk)
            rms = np.sqrt(np.mean(chunk ** 2)) * 32767

            if rms > self.rms_command:
                started = True
                silent = 0
            elif started:
                silent += 1
                if silent >= silence_chunks:
                    break

        self.get_logger().info(f'[CMD] done — total:{len(buf)} chunks, voice:{started}')
        return np.concatenate(buf) if buf else np.array([], dtype=np.float32)

    def _listen_loop(self):
        WAKE_WORDS = [
            'vector', 'vektor', 'hector', 'victor', 'vectra',
            'specter', 'vactor', 'wector', 'picture', 'with that',
        ]

        while not self._stop_event.is_set() and rclpy.ok():
            if not self.audio_enabled:
                try:
                    self.audio_q.get(timeout=0.1)
                except queue.Empty:
                    pass
                continue

            if self.is_tts_speaking:
                try:
                    self.audio_q.get(timeout=0.1)
                except queue.Empty:
                    pass
                continue

            try:
                chunk = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if self.is_tts_speaking:
                continue

            rms = np.sqrt(np.mean(chunk ** 2)) * 32767
            if rms < self.rms_wake:
                continue

            pre = [chunk]
            for _ in range(14):
                if self.is_tts_speaking:
                    break
                try:
                    pre.append(self.audio_q.get(timeout=0.15))
                except queue.Empty:
                    break

            if self.is_tts_speaking:
                continue

            audio = np.concatenate(pre)
            text = self._transcribe(audio, prompt='Vector')
            self.get_logger().info(f'[Wake] heard: "{text}"')

            if not text.strip():
                continue

            words = text.split()
            if words and words.count(max(set(words), key=words.count)) > 4:
                continue

            if not any(w in text for w in WAKE_WORDS):
                continue

            self.get_logger().info('✅ Vector! Listening for command...')
            t_wake_pub = time.time()
            self._publish('WAKE_WORD_DETECTED')

            self._wait_for_ack_tts()
            t_ack_done = time.time()
            self.get_logger().info(
                f'[TIMING] wake_pub → ack_done: {t_ack_done - t_wake_pub:.2f}s'
            )

            cmd_audio = self._collect_until_silence(max_sec=self.command_max_sec)

            if self.is_tts_speaking:
                continue

            if cmd_audio.size == 0:
                self.get_logger().info('[CMD] No audio captured — resuming')
                self._publish('RESUME')
                continue

            cmd_rms = np.sqrt(np.mean(cmd_audio ** 2)) * 32767
            if cmd_rms < self.rms_command:
                self.get_logger().info(f'[CMD] Too quiet (rms={cmd_rms:.0f}) — resuming')
                self._publish('RESUME')
                continue

            cmd_text = self._transcribe(
                cmd_audio,
                prompt='follow me, follow, stop, forward, backward, left, right, search, describe the scene, battery, battery status, battery level, battery percentage, power level, charge level'
            )
            if not cmd_text.strip():
                self.get_logger().info('[CMD] Empty transcription — resuming')
                self._publish('RESUME')
                continue

            command = self._parse(cmd_text)
            self.get_logger().info(f'[CMD] "{cmd_text}" → {command}')

            if command == 'NONE':
                self.get_logger().info(f'[CMD] Sending to LLM: "{cmd_text}"')
                self._publish(cmd_text)
            else:
                self._publish(command)

            self._flush_queue()
            time.sleep(0.5)
            self._flush_queue()
            self.get_logger().info('🔄 Listening for "Vector"...')

    def _publish(self, data: str):
        msg = String()
        msg.data = data
        self.pub.publish(msg)

    def _parse(self, text: str) -> str:
        t = text.lower()
        if any(x in t for x in ['follow me', 'follow person', 'come with me', 'follow']):
            return 'FOLLOW_PERSON'
        elif any(x in t for x in ['stop', 'halt', 'freeze']):
            return 'STOP'
        elif any(x in t for x in ['search', 'find person', 'look for']):
            return 'SEARCH'
        elif any(x in t for x in ['turn left', 'left']):
            return 'LEFT'
        elif any(x in t for x in ['turn right', 'right']):
            return 'RIGHT'
        elif any(x in t for x in ['move forward', 'go ahead', 'forward']):
            return 'FORWARD'
        elif any(x in t for x in ['move backward', 'go back', 'backward']):
            return 'BACKWARD'
        elif any(x in t for x in [
            'battery', 'batteries', 'batery', 'battery status', 'battery level',
            'battery percentage', 'remaining battery', 'power level', 'power status',
            'charge level', 'remaining charge', 'how much charge',
        ]):
            return 'BATTERY_STATUS'
        elif any(x in t for x in [
            'describe the scene', 'describe scene', 'describe the seen', 'describe seen',
            'what do you see', 'what can you see', 'whats around', "what's around",
            'look around', 'describe what you see',
        ]) or 'describe' in t:
            return 'DESCRIBE_SCENE'
        return 'NONE'


def main(args=None):
    rclpy.init(args=args)
    node = STTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if rclpy.ok():
            node.get_logger().info('STT node stopping...')
    finally:
        node._stop_event.set()
        node._listen_thread.join(timeout=3.0)
        node.stream.stop()
        node.stream.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()