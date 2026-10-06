import time
import io
import soundfile as sf
import numpy as np
from faster_whisper import WhisperModel

print("Loading Whisper model...")
start = time.time()
model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
load_time = time.time() - start
print(f"Model load time: {load_time:.2f}s")

print("Generating 2 seconds of silence (fake audio)...")
sample_rate = 16000
data = np.zeros(sample_rate * 2, dtype=np.float32)

clean_wav_io = io.BytesIO()
sf.write(clean_wav_io, data, sample_rate, subtype='PCM_16', format='WAV')
clean_wav_io.seek(0)

print("Transcribing...")
start = time.time()
segments, info = model.transcribe(clean_wav_io, language="en", beam_size=5)
transcript = "".join(segment.text for segment in segments).strip()
transcribe_time = time.time() - start

print(f"Transcription time: {transcribe_time:.2f}s")
print(f"Transcript: '{transcript}'")
