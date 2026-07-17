import time
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import whisper
import sounddevice as sd
import numpy as np
import tempfile
import os
import threading
import queue
from scipy.io.wavfile import write as wav_write

SAMPLE_RATE  = 16000   # Whisper native rate — record directly at 16kHz
CHUNK_SIZE   = 1600    # 100ms chunks
RMS_WAKE     = 800     # RMS above this = possible speech (tune if needed)
SILENCE_SEC  = 1.0     # seconds of silence to end command capture

class STTNode(Node):
    def __init__(self):
        super().__init__('stt_node')
        self.pub = self.create_publisher(String, '/voice_command', 10)

        self.get_logger().info('Loading Whisper...')
        self.model = whisper.load_model('tiny.en').to('cuda')

        self.audio_q = queue.Queue()
        self.stream  = sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype='float32',
            blocksize=CHUNK_SIZE,
            callback=lambda i, f, t, s: self.audio_q.put(i.copy().flatten())
        )
        self.stream.start()

        threading.Thread(target=self._listen_loop, daemon=True).start()
        self.get_logger().info('★ Ready. Say "Vector".')

    # ── Transcribe a numpy float32 array ──────────────────

    def _transcribe(self, audio: np.ndarray, prompt: str) -> str:
        tmp = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        wav_write(tmp.name, SAMPLE_RATE, (audio * 32767).astype(np.int16))
        tmp.close()
        try:
            r = self.model.transcribe(
                tmp.name, fp16=True, language='en',
                initial_prompt=prompt,
                best_of=1, beam_size=1, temperature=0.0,
                no_speech_threshold=0.6,
            )
            return r['text'].lower().strip()
        finally:
            os.remove(tmp.name)

    # ── Collect audio until RMS silence ───────────────────

    def _collect_until_silence(self, max_sec=5.0) -> np.ndarray:
        silence_chunks = int(SILENCE_SEC * SAMPLE_RATE / CHUNK_SIZE)
        max_chunks     = int(max_sec    * SAMPLE_RATE / CHUNK_SIZE)
        buf, silent, started = [], 0, False

        for _ in range(max_chunks):
            try:
                chunk = self.audio_q.get(timeout=0.3)
            except queue.Empty:
                break
            buf.append(chunk)
            rms = np.sqrt(np.mean(chunk ** 2)) * 32767
            if rms > RMS_WAKE:
                started = True
                silent  = 0
            elif started:
                silent += 1
                if silent >= silence_chunks:
                    break

        return np.concatenate(buf) if buf else np.array([], dtype=np.float32)

    # ── Main loop ─────────────────────────────────────────

    def _listen_loop(self):
        WAKE_WORDS = [
            'vector','vektor','hector','victor','vectra',
            'specter','vactor','wector','picture','with that',
        ]
        silence_chunks = int(0.3 * SAMPLE_RATE / CHUNK_SIZE)  # 300ms pre-trigger gate

        while rclpy.ok():
            # Step 1: Wait for RMS spike (cheap, no model)
            try:
                chunk = self.audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            rms   = np.sqrt(np.mean(chunk ** 2)) * 32767
            if rms < RMS_WAKE:
                continue

            # Step 2: Collect ~1.5s around the spike for wake word check
            pre = [chunk]
            for _ in range(14):   # 14 × 100ms = 1.4s more
                try:
                    pre.append(self.audio_q.get(timeout=0.15))
                except queue.Empty:
                    break

            audio = np.concatenate(pre)
            text  = self._transcribe(audio, prompt='Vector')
            self.get_logger().info(f'[Wake] heard: "{text}"')

            if not text.strip():          
                continue

            # Hallucination check
            words = text.split()
            if words and words.count(max(set(words), key=words.count)) > 4:
                continue

            if not any(w in text for w in WAKE_WORDS):
                continue

            # Step 3: Wake confirmed — publish and open command window
            self.get_logger().info('✅ Vector! Listening for command...')
            self._publish('WAKE_WORD_DETECTED') 
            time.sleep(0.3)

            # Drain stale audio from queue
            while not self.audio_q.empty():
                self.audio_q.get_nowait()

            # Step 4: Collect command (VAD by RMS, ends on silence)
            cmd_audio = self._collect_until_silence(max_sec=4.0)
            
            cmd_rms = np.sqrt(np.mean(cmd_audio ** 2)) * 32767 if cmd_audio.size > 0 else 0
            if cmd_audio.size == 0 or cmd_rms < RMS_WAKE:
                self.get_logger().info(f'[CMD] No command heard — resuming')
                self._publish('RESUME')  # ← tells decision node to go back to AUTO
                continue

            cmd_text = self._transcribe(
                cmd_audio,
                prompt='follow me, stop, forward, backward, left, right, search'
            )
            command = self._parse(cmd_text)
            self.get_logger().info(f'[CMD] "{cmd_text}" → {command}')
            self._publish(command)

            # Drain leftover command audio so it doesn't re-trigger wake word
            while not self.audio_q.empty():
                self.audio_q.get_nowait()

            time.sleep(0.8)
            
            while not self.audio_q.empty():
                self.audio_q.get_nowait()
            # ── END ADD ──

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
        return 'NONE'


def main(args=None):
    rclpy.init(args=args)
    node = STTNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stream.stop()
        node.stream.close()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()