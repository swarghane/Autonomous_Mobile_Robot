import os
os.environ.setdefault('XDG_RUNTIME_DIR', '/run/user/1000')

import asyncio
import threading
import time
from pathlib import Path

import numpy as np
import rclpy
import sounddevice as sd
from piper import PiperVoice, SynthesisConfig
from rclpy.node import Node
from std_msgs.msg import Bool, Float32, String


class TTSNode(Node):
    """Offline Piper TTS with persistent audio playback and mouth lip-sync."""

    SUPPORTED_VOICES = {
        'en_US-hfc_male-medium',
        'en_US-hfc_female-medium',
    }

    VOICE_PROFILES = {
        'en_US-hfc_male-medium': {
            'length_scale': 0.94,
            'noise_scale': 0.55,
            'noise_w_scale': 0.70,
        },
        'en_US-hfc_female-medium': {
            'length_scale': 1.05,
            'noise_scale': 0.62,
            'noise_w_scale': 0.75,
        },
    }

    def __init__(self):
        super().__init__('tts_node')

        self.declare_parameter('piper_voice', 'en_US-hfc_male-medium')
        self.declare_parameter('piper_data_dir', '/workspace/models/piper')
        self.declare_parameter('use_voice_profile', True)
        self.declare_parameter('length_scale', 0.94)
        self.declare_parameter('noise_scale', 0.55)
        self.declare_parameter('noise_w_scale', 0.70)
        self.declare_parameter('volume', 1.0)
        self.declare_parameter('normalize_audio', True)

        self.declare_parameter('tts_status_topic', '/tts_status')
        self.declare_parameter('robot_emotion_topic', '/robot_emotion')
        self.declare_parameter('mouth_level_topic', '/mouth_level')
        self.declare_parameter('llm_response_topic', '/llm_response')
        self.declare_parameter('voice_command_topic', '/voice_command')
        self.declare_parameter('audio_enabled_topic', '/audio_enabled')

        self.declare_parameter('chunk_ms', 50)
        self.declare_parameter('bt_latency_s', 0.1)
        self.declare_parameter('post_playback_delay_s', 0.4)
        self.declare_parameter('post_playback_delay_cached_s', 0.1)
        self.declare_parameter('voice_command_dedupe_s', 2.0)
        self.declare_parameter('startup_speech_delay_s', 5.0)
        self.declare_parameter('startup_announcement_enabled', True)
        self.declare_parameter('cache_fixed_phrases', True)

        self.declare_parameter('audio_samplerate', 44100)
        self.declare_parameter('bt_keepalive_enabled', True)
        self.declare_parameter('bt_keepalive_volume', 0.003)

        self.piper_voice_name = self.get_parameter('piper_voice').value
        self.piper_data_dir = Path(self.get_parameter('piper_data_dir').value)
        self.use_voice_profile = bool(self.get_parameter('use_voice_profile').value)
        self.length_scale = float(self.get_parameter('length_scale').value)
        self.noise_scale = float(self.get_parameter('noise_scale').value)
        self.noise_w_scale = float(self.get_parameter('noise_w_scale').value)

        self._validate_voice_selection()
        if self.use_voice_profile:
            profile = self.VOICE_PROFILES[self.piper_voice_name]
            self.length_scale = profile['length_scale']
            self.noise_scale = profile['noise_scale']
            self.noise_w_scale = profile['noise_w_scale']
        self.volume = float(self.get_parameter('volume').value)
        self.normalize_audio = bool(self.get_parameter('normalize_audio').value)

        self.tts_status_topic = self.get_parameter('tts_status_topic').value
        self.robot_emotion_topic = self.get_parameter('robot_emotion_topic').value
        self.mouth_level_topic = self.get_parameter('mouth_level_topic').value
        self.llm_response_topic = self.get_parameter('llm_response_topic').value
        self.voice_command_topic = self.get_parameter('voice_command_topic').value
        self.audio_enabled_topic = self.get_parameter('audio_enabled_topic').value

        self.chunk_ms = int(self.get_parameter('chunk_ms').value)
        self.bt_latency_s = float(self.get_parameter('bt_latency_s').value)
        self.post_playback_delay_s = float(self.get_parameter('post_playback_delay_s').value)
        self.post_playback_delay_cached_s = float(self.get_parameter('post_playback_delay_cached_s').value)
        self._voice_cmd_dedupe_s = float(self.get_parameter('voice_command_dedupe_s').value)

        configured_sample_rate = int(self.get_parameter('audio_samplerate').value)
        self.bt_keepalive_enabled = bool(self.get_parameter('bt_keepalive_enabled').value)
        self.bt_keepalive_volume = float(self.get_parameter('bt_keepalive_volume').value)
        self.startup_speech_delay_s = float(self.get_parameter('startup_speech_delay_s').value)
        self.startup_announcement_enabled = bool(self.get_parameter('startup_announcement_enabled').value)
        self.cache_fixed_phrases = bool(self.get_parameter('cache_fixed_phrases').value)

        self.piper_voice = self._load_piper_voice()
        self.piper_sample_rate = int(self.piper_voice.config.sample_rate)
        self.audio_samplerate = configured_sample_rate or self.piper_sample_rate

        self.synthesis_config = SynthesisConfig(
            length_scale=self.length_scale,
            noise_scale=self.noise_scale,
            noise_w_scale=self.noise_w_scale,
            volume=self.volume,
            normalize_audio=self.normalize_audio,
        )

        self.status_pub = self.create_publisher(String, self.tts_status_topic, 10)
        self.emotion_pub = self.create_publisher(String, self.robot_emotion_topic, 10)
        self.mouth_pub = self.create_publisher(Float32, self.mouth_level_topic, 10)

        self.loop = asyncio.new_event_loop()
        self.speech_q = None

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
        self.cacheable_texts = set(self.cached_phrases.values())
        self.cache_decoded = {}

        self._last_voice_cmd = None
        self._last_voice_cmd_time = 0.0
        self._last_audio_enabled = None

        self._audio_lock = threading.Lock()
        self._synthesis_lock = threading.Lock()
        self._playback_buffer = None
        self._playback_pos = 0
        self._tone_phase = 0
        self._audio_stream = None

        threading.Thread(target=self._start_audio_engine, daemon=True).start()
        threading.Thread(target=self._start_async_loop, daemon=True).start()

        self.create_subscription(String, self.llm_response_topic, self.response_callback, 10)
        self.create_subscription(String, self.voice_command_topic, self.command_callback, 10)
        self.create_subscription(Bool, self.audio_enabled_topic, self._audio_enabled_callback, 10)

        self._publish_status('IDLE')
        self._publish_emotion('neutral')
        self._publish_mouth(0.0)

        self.get_logger().info(
            f'Piper TTS ready: voice={self.piper_voice_name}, '
            f'model_rate={self.piper_sample_rate} Hz, '
            f'output_rate={self.audio_samplerate} Hz, '
            f'lazy_cache={self.cache_fixed_phrases}'
        )

    def _validate_voice_selection(self):
        if self.piper_voice_name not in self.SUPPORTED_VOICES:
            supported = ', '.join(sorted(self.SUPPORTED_VOICES))
            raise ValueError(
                f'Unsupported piper_voice "{self.piper_voice_name}". '
                f'Supported voices: {supported}'
            )

    def _load_piper_voice(self):
        model_path = self.piper_data_dir / f'{self.piper_voice_name}.onnx'
        config_path = Path(f'{model_path}.json')

        if not model_path.is_file():
            raise FileNotFoundError(
                f'Piper model not found: {model_path}. Download it with: '
                f'python3 -m piper.download_voices --data-dir '
                f'{self.piper_data_dir} {self.piper_voice_name}'
            )
        if not config_path.is_file():
            raise FileNotFoundError(f'Piper config not found: {config_path}')

        self.get_logger().info(f'Loading Piper model: {model_path}')
        return PiperVoice.load(str(model_path), use_cuda=False)

    def _keepalive_chunk(self, frames: int) -> np.ndarray:
        volume = self.bt_keepalive_volume if self.bt_keepalive_enabled else 0.0
        frequency = 40.0
        t = (self._tone_phase + np.arange(frames)) / self.audio_samplerate
        self._tone_phase += frames
        return (volume * np.sin(2 * np.pi * frequency * t)).astype(np.float32)

    def _audio_callback(self, outdata, frames, time_info, status):
        del time_info, status

        with self._audio_lock:
            buffer = self._playback_buffer
            position = self._playback_pos

        if buffer is not None and position < len(buffer):
            end = min(position + frames, len(buffer))
            sample_count = end - position
            outdata[:sample_count, 0] = buffer[position:end]

            if sample_count < frames:
                outdata[sample_count:, 0] = self._keepalive_chunk(frames - sample_count)

            with self._audio_lock:
                self._playback_pos = end
        else:
            outdata[:, 0] = self._keepalive_chunk(frames)

    def _start_audio_engine(self):
        try:
            self._audio_stream = sd.OutputStream(
                samplerate=self.audio_samplerate,
                channels=1,
                dtype='float32',
                callback=self._audio_callback,
            )
            self._audio_stream.start()
            self.get_logger().info(
                f'[TTS] Persistent audio engine started '
                f'(keepalive={self.bt_keepalive_enabled}, '
                f'volume={self.bt_keepalive_volume})'
            )
        except Exception as error:
            self.get_logger().error(f'[TTS] Failed to start audio engine: {error}')

    def _stop_playback(self):
        with self._audio_lock:
            self._playback_buffer = None
            self._playback_pos = 0

    def _synthesize_text(self, text):
        chunks = []
        source_sample_rate = None

        with self._synthesis_lock:
            for chunk in self.piper_voice.synthesize(text, syn_config=self.synthesis_config):
                source_sample_rate = int(chunk.sample_rate)
                pcm = np.frombuffer(chunk.audio_int16_bytes, dtype=np.int16)

                channels = int(chunk.sample_channels)
                if channels > 1:
                    pcm = pcm.reshape(-1, channels).mean(axis=1)

                chunks.append(pcm.astype(np.float32) / 32768.0)

        if not chunks or source_sample_rate is None:
            raise RuntimeError('Piper returned no audio samples')

        samples = np.concatenate(chunks)
        if source_sample_rate != self.audio_samplerate:
            samples = self._resample_audio(samples, source_sample_rate, self.audio_samplerate)

        levels = self._calculate_lipsync_levels(samples)
        return samples, levels

    @staticmethod
    def _resample_audio(samples, source_rate, target_rate):
        if source_rate == target_rate or len(samples) < 2:
            return samples.astype(np.float32, copy=False)

        target_length = max(1, int(round(len(samples) * target_rate / source_rate)))
        source_positions = np.arange(len(samples), dtype=np.float64)
        target_positions = np.linspace(0, len(samples) - 1, target_length, dtype=np.float64)
        return np.interp(target_positions, source_positions, samples).astype(np.float32)

    def _calculate_lipsync_levels(self, samples):
        chunk_samples = max(1, int(self.audio_samplerate * self.chunk_ms / 1000))
        raw_levels = []

        for index in range(0, len(samples), chunk_samples):
            window = samples[index:index + chunk_samples]
            rms = float(np.sqrt(np.mean(window * window))) if len(window) else 0.0
            raw_levels.append(rms)

        max_rms = max(raw_levels) if raw_levels else 1.0
        if max_rms <= 0.0:
            max_rms = 1.0

        return [min(1.0, rms / max_rms) for rms in raw_levels]

    def response_callback(self, msg):
        text = msg.data.strip()
        if text and self.speech_q:
            self.loop.call_soon_threadsafe(self.speech_q.put_nowait, text)

    def command_callback(self, msg):
        command = msg.data.strip().upper()
        now = time.time()

        if command == self._last_voice_cmd and now - self._last_voice_cmd_time < self._voice_cmd_dedupe_s:
            self.get_logger().info(f'[TTS] Duplicate command ignored: "{command}"')
            return

        self._last_voice_cmd = command
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
        text = confirmations.get(command)

        if text and self.speech_q:
            self.loop.call_soon_threadsafe(self.speech_q.put_nowait, text)

    def _audio_enabled_callback(self, msg):
        enabled = msg.data
        if enabled == self._last_audio_enabled:
            return

        self._last_audio_enabled = enabled
        text = 'Voice commands on.' if enabled else 'Voice commands off.'

        if self.speech_q:
            self.loop.call_soon_threadsafe(self.speech_q.put_nowait, text)

    def _publish_status(self, text):
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def _publish_emotion(self, text):
        msg = String()
        msg.data = text
        self.emotion_pub.publish(msg)

    def _publish_mouth(self, level):
        msg = Float32()
        msg.data = max(0.0, min(1.0, float(level)))
        self.mouth_pub.publish(msg)

    def _start_async_loop(self):
        asyncio.set_event_loop(self.loop)
        self.speech_q = asyncio.Queue()
        self.loop.run_until_complete(self._speech_loop())

    async def _lipsync_stream(self, levels, start_delay=0.0):
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

    async def _queue_startup_message(self):
        await asyncio.sleep(self.startup_speech_delay_s)
        if rclpy.ok() and self.speech_q is not None:
            await self.speech_q.put('Vector is ready.')

    async def _speech_loop(self):
        if self.startup_announcement_enabled:
            asyncio.create_task(self._queue_startup_message())
        while rclpy.ok():
            text = await self.speech_q.get()
            if not text:
                self.speech_q.task_done()
                continue

            lipsync_task = None
            is_cached = text in self.cache_decoded

            try:
                start_time = time.time()
                self._publish_status('SPEAKING')
                self.get_logger().info(f'TTS: {text}')

                if is_cached:
                    samples, levels = self.cache_decoded[text]
                else:
                    loop = asyncio.get_running_loop()
                    samples, levels = await loop.run_in_executor(None, self._synthesize_text, text)
                    if self.cache_fixed_phrases and text in self.cacheable_texts:
                        self.cache_decoded[text] = (samples, levels)
                        self.get_logger().info(f'[TTS] Cached after first use: "{text}"')

                synthesis_done = time.time()
                self.get_logger().info(
                    f'[TIMING] Piper synthesis (cached={is_cached}): {synthesis_done - start_time:.2f}s'
                )

                self._publish_emotion('talking')

                with self._audio_lock:
                    self._playback_buffer = samples
                    self._playback_pos = 0

                if levels:
                    lipsync_task = asyncio.create_task(
                        self._lipsync_stream(levels, start_delay=self.bt_latency_s)
                    )

                duration = len(samples) / self.audio_samplerate
                await asyncio.sleep(duration)
                playback_done = time.time()
                self.get_logger().info(f'[TIMING] Playback: {playback_done - synthesis_done:.2f}s')

                self._stop_playback()
                delay = self.post_playback_delay_cached_s if is_cached else self.post_playback_delay_s
                await asyncio.sleep(delay)

                self.get_logger().info(f'[TIMING] TOTAL SPEAKING to IDLE: {time.time() - start_time:.2f}s')
            except Exception as error:
                self.get_logger().error(f'[TTS] {error}')
            finally:
                if lipsync_task:
                    lipsync_task.cancel()

                self._publish_mouth(0.0)
                self._stop_playback()
                self._publish_status('IDLE')
                self._publish_emotion('neutral')
                self.speech_q.task_done()

    async def stop_speaking(self):
        self._stop_playback()

    def destroy_node(self):
        self._stop_playback()

        if self._audio_stream is not None:
            try:
                self._audio_stream.stop()
                self._audio_stream.close()
            except Exception as error:
                self.get_logger().warning(f'[TTS] Audio stream shutdown warning: {error}')

        if self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
        elif not self.loop.is_closed():
            self.loop.close()

        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = None

    try:
        node = TTSNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        if node is not None and rclpy.ok():
            node.get_logger().info('TTS node stopping...')
    except Exception as error:
        if node is not None:
            node.get_logger().fatal(f'TTS node failed: {error}')
        else:
            print(f'TTS node failed: {error}')
        raise
    finally:
        if node is not None:
            node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()