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

# ─── Audio config ───────────────────────────────────────────────
SAMPLE_RATE   = 44100   # USB mic native rate — no ALSA resampling errors
CHUNK_SIZE    = 4410    # 100ms chunks
WHISPER_RATE  = 16000   # Whisper always needs 16kHz

RMS_WAKE      = 600     # threshold to detect any speech
RMS_COMMAND   = 150     # lower threshold — command capture is more sensitive
SILENCE_SEC   = 1.5     # seconds of silence before command is considered done
# ────────────────────────────────────────────────────────────────

class STTNode(Node):
    def __init__(self):
        super().__init__('stt_node')
        self._stop_event = threading.Event()

        self.pub = self.create_publisher(String, '/voice_command', 10)

        self.create_subscription(
            String, '/tts_status',
            self._tts_status_callback, 10
        )
        self.is_tts_speaking = False

        # Audio ON/OFF toggle (webpage button or raised-hand gesture).
        self.audio_enabled = True
        self.create_subscription(
            Bool, '/audio_enabled',
            self._audio_enabled_callback, 10
        )

        self.get_logger().info('Loading Whisper (faster-whisper, int8)...')
        # int8_float16: int8 weights, float16 compute — good speed/accuracy
        # balance on Jetson GPUs. Try "int8" (pure int8) if you want to
        # push speed further, or "float16" if accuracy matters more.
        self.model = WhisperModel('tiny.en', device='cuda', compute_type='int8_float16')

        self.audio_q = queue.Queue()
        self.stream  = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=CHUNK_SIZE,
            # device='USB PnP Sound Device',
            callback=lambda i, f, t, s: self.audio_q.put(i.copy().flatten())
        )
        self.stream.start()

        self._stop_event = threading.Event()
        self._listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self._listen_thread.start()
        self.get_logger().info('★ Ready. Say "Vector".')

    # ─── TTS mute / unmute ──────────────────────────────────────
    def _tts_status_callback(self, msg: String):
        status = msg.data.upper()
        if status == 'SPEAKING':
            self.is_tts_speaking = True
            self.get_logger().info('🔇 TTS Active: muting STT.')
        elif status == 'IDLE':
            time.sleep(0.6)          # let room echo decay
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

    def _wait_for_ack_tts(self, start_timeout=0.8, total_timeout=8.0):
        """
        After publishing WAKE_WORD_DETECTED, tts_node speaks a short "Yes?"
        acknowledgment. Wait for that to actually start (SPEAKING) and then
        finish (IDLE) before opening the real command-listening window —
        otherwise _collect_until_silence races the ack and gets killed by
        the is_tts_speaking mute check within milliseconds.
        """
        start = time.time()
        while not self.is_tts_speaking and (time.time() - start) < start_timeout:
            time.sleep(0.02)
        while self.is_tts_speaking and (time.time() - start) < total_timeout:
            time.sleep(0.02)

    # ─── Resample + transcribe ──────────────────────────────────
    def _to_whisper(self, audio: np.ndarray) -> np.ndarray:
        """Resample from 44100 → 16000 for Whisper."""
        return resample_poly(audio, WHISPER_RATE, SAMPLE_RATE).astype(np.float32)

    def _transcribe(self, audio: np.ndarray, prompt: str) -> str:
        audio_16k = self._to_whisper(audio)

        # faster-whisper accepts a numpy float32 array directly — no need
        # to round-trip through a temp WAV file like openai-whisper required.
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
            # Hallucination guard — compression ratio (same check as before)
            if getattr(seg, 'compression_ratio', 0) > 2.4:
                self.get_logger().warn('[STT] Hallucination detected — ignored')
                return ''
            text_parts.append(seg.text)

        text = ''.join(text_parts).lower().strip()

        # Hallucination guard — word repetition loop
        words = text.split()
        if len(words) > 5:
            top = max(set(words), key=words.count)
            if words.count(top) / len(words) > 0.5:
                self.get_logger().warn(f'[STT] Repetition loop on "{top}" — ignored')
                return ''

        return text

    # ─── Collect audio until silence ────────────────────────────
    def _collect_until_silence(self, max_sec=6.0, prefill=None) -> np.ndarray:
        """
        Collect audio chunks until silence or timeout.
        prefill: list of chunks already captured (e.g. tail of wake word audio)
        """
        silence_chunks = int(SILENCE_SEC * SAMPLE_RATE / CHUNK_SIZE)
        max_chunks     = int(max_sec     * SAMPLE_RATE / CHUNK_SIZE)

        # Seed buffer with any audio already captured
        buf     = list(prefill) if prefill else []
        silent  = 0
        started = False

        # Check if prefill already contains voice
        for chunk in buf:
            rms = np.sqrt(np.mean(chunk ** 2)) * 32767
            if rms > RMS_COMMAND:
                started = True
                break

        self.get_logger().info(
            f'[CMD] collecting — prefill:{len(buf)} chunks, '
            f'voice_detected:{started}, threshold:{RMS_COMMAND}'
        )

        remaining = max_chunks - len(buf)
        for _ in range(remaining):
            if self.is_tts_speaking or not self.audio_enabled:
                return np.array([], dtype=np.float32)

            try:
                chunk = self.audio_q.get(timeout=0.3)
            except queue.Empty:
                # If we already heard voice, silence counts as end of speech
                if started:
                    silent += 1
                    if silent >= silence_chunks:
                        break
                continue

            buf.append(chunk)
            rms = np.sqrt(np.mean(chunk ** 2)) * 32767

            if rms > RMS_COMMAND:
                started = True
                silent  = 0
            elif started:
                silent += 1
                if silent >= silence_chunks:
                    break

        self.get_logger().info(f'[CMD] done — total:{len(buf)} chunks, voice:{started}')
        return np.concatenate(buf) if buf else np.array([], dtype=np.float32)

    # ─── Main listen loop ───────────────────────────────────────
    def _listen_loop(self):
        WAKE_WORDS = [
            'vector', 'vektor', 'hector', 'victor', 'vectra',
            'specter', 'vactor', 'wector', 'picture', 'with that',
        ]

        while not self._stop_event.is_set() and rclpy.ok():

            # Audio mode OFF (webpage toggle / gesture) — drain mic input
            # and skip wake-word detection entirely until re-enabled.
            if not self.audio_enabled:
                try:
                    self.audio_q.get(timeout=0.1)
                except queue.Empty:
                    pass
                continue

            # Drain queue while TTS is speaking
            if self.is_tts_speaking:
                try:
                    self.audio_q.get(timeout=0.1)
                except queue.Empty:
                    pass
                continue

            # ── Step 1: wait for any RMS spike ──────────────────
            try:
                chunk = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            if self.is_tts_speaking:
                continue

            rms = np.sqrt(np.mean(chunk ** 2)) * 32767
            if rms < RMS_WAKE:
                continue

            # ── Step 2: collect ~1.5s around the spike ──────────
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

            # ── Step 3: check for wake word ──────────────────────
            audio = np.concatenate(pre)
            text  = self._transcribe(audio, prompt='Vector')
            self.get_logger().info(f'[Wake] heard: "{text}"')

            if not text.strip():
                continue

            words = text.split()
            if words and words.count(max(set(words), key=words.count)) > 4:
                continue

            if not any(w in text for w in WAKE_WORDS):
                continue

            # ── Step 4: wake confirmed — collect command ─────────
            self.get_logger().info('✅ Vector! Listening for command...')
            self._publish('WAKE_WORD_DETECTED')

            # Let the "Yes?" acknowledgment fully play out first — starting
            # command collection immediately used to race the TTS mute
            # check and get killed within milliseconds of "Yes?" starting.
            self._wait_for_ack_tts()

            # Buffer is already flushed by _tts_status_callback's IDLE
            # handler, so prefill is stale/irrelevant here — start fresh.
            cmd_audio = self._collect_until_silence(max_sec=6.0)

            if self.is_tts_speaking:
                continue

            if cmd_audio.size == 0:
                self.get_logger().info('[CMD] No audio captured — resuming')
                self._publish('RESUME')
                continue

            cmd_rms = np.sqrt(np.mean(cmd_audio ** 2)) * 32767
            if cmd_rms < RMS_COMMAND:
                self.get_logger().info(f'[CMD] Too quiet (rms={cmd_rms:.0f}) — resuming')
                self._publish('RESUME')
                continue

            # ── Step 5: transcribe command ───────────────────────
            cmd_text = self._transcribe(
                cmd_audio,
                prompt='follow me, stop, forward, backward, left, right, search, describe the scene'
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

            # ── Step 6: clean up before next wake cycle ──────────
            self._flush_queue()
            time.sleep(0.5)
            self._flush_queue()
            self.get_logger().info('🔄 Listening for "Vector"...')

    # ─── Helpers ────────────────────────────────────────────────
    def _publish(self, data: str):
        msg      = String()
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