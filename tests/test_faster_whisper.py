import time
from faster_whisper import WhisperModel

start = time.time()

print("Loading model...")
model = WhisperModel(
    "tiny.en",
    device="cuda",
    compute_type="int8_float16"
)

print(f"Load time: {time.time()-start:.2f}s")

start = time.time()

segments, info = model.transcribe(
    "test.wav",
    beam_size=1,
)

print(f"Inference time: {time.time()-start:.2f}s")

for s in segments:
    print(s.text)