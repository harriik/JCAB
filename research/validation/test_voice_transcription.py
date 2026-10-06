import os
import sys
import numpy as np
import soundfile as sf
import io
import time

# Create a dummy silent WAV just to test the pipeline executes
# In a real CI environment, we would load a fixture.
data = np.zeros(16000, dtype='float32') # 1 second of silence
wav_io = io.BytesIO()
sf.write(wav_io, data, 16000, format='WAV')
wav_bytes = wav_io.getvalue()

# We need to import the function from app.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from app import transcribe_voice

def test_transcription():
    print("Testing transcribe_voice pipeline...")
    success, transcript, err_type, err_msg, duration, latency = transcribe_voice(wav_bytes)
    
    print(f"Success: {success}")
    print(f"Transcript: '{transcript}'")
    print(f"Error Type: {err_type}")
    print(f"Error Msg: {err_msg}")
    print(f"Duration: {duration:.2f}s")
    print(f"Latency: {latency:.2f}s")
    
    assert duration == 1.0, "Duration should be exactly 1.0s"
    # Silence is transcribed as "You" or empty by Whisper
    print("Pipeline executed successfully without crashing.")

if __name__ == "__main__":
    test_transcription()
