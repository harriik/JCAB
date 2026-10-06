import streamlit as st
import numpy as np
import soundfile as sf
import io
import os
import pkg_resources

st.set_page_config(page_title="Mic Capture Test", page_icon="🎤")

st.title("Microphone Capture Diagnostic")

try:
    version = pkg_resources.get_distribution("streamlit-mic-recorder").version
    st.write(f"**Installed streamlit-mic-recorder version:** {version}")
except Exception as e:
    st.error(f"Could not check version: {e}")

st.info("⚠️ Ensure you click **Allow** when Chrome asks for microphone permissions, and check that the correct physical microphone is selected in Chrome settings.")

from streamlit_mic_recorder import mic_recorder

st.subheader("Microphone Capture Test")

audio_val = mic_recorder(
    start_prompt="🎤 Start Recording Test", 
    stop_prompt="⏹️ Stop Recording", 
    key="mic_test", 
    format="wav"
)

if audio_val is not None:
    audio_bytes = audio_val['bytes']
    raw_size = len(audio_bytes)
    st.write(f"**Browser Recorder Status:** Success (Returned Bytes)")
    st.write(f"**Raw Byte Size:** {raw_size} bytes")
    
    if raw_size > 0:
        # Save raw binary
        raw_path = r"D:\JCAB\research\validation\mic_raw_capture.bin"
        with open(raw_path, "wb") as f:
            f.write(audio_bytes)
        st.success(f"Saved raw bytes to: `{raw_path}`")
        
        # Decode
        try:
            with sf.SoundFile(io.BytesIO(audio_bytes)) as sfile:
                channels = sfile.channels
                samplerate = sfile.samplerate
                subtype = sfile.subtype
                file_format = sfile.format
                frames = sfile.frames
                
                data = sfile.read(dtype='float32')
                
                st.write("**WAV Metadata:**")
                st.json({
                    "channels": channels,
                    "sample_rate": samplerate,
                    "subtype": subtype,
                    "format": file_format,
                    "frames": frames
                })
                
                if len(data.shape) > 1:
                    data = data.mean(axis=1) # mono
                    
                duration = len(data) / samplerate
                min_amp = float(np.min(data))
                max_amp = float(np.max(data))
                mean_abs = float(np.mean(np.abs(data)))
                rms = float(np.sqrt(np.mean(data**2)))
                nonzero_pct = float(np.mean(data != 0)) * 100
                
                st.write("**Audio Diagnostics:**")
                st.json({
                    "duration_seconds": round(duration, 2),
                    "min_amplitude": round(min_amp, 6),
                    "max_amplitude (peak)": round(max_amp, 6),
                    "mean_absolute_amplitude": round(mean_abs, 6),
                    "RMS": round(rms, 6),
                    "nonzero_percentage": round(nonzero_pct, 2)
                })
                
                if rms < 0.0005:
                    st.error("🚨 **SIGNAL IS SILENT** (RMS effectively zero)")
                else:
                    st.success("✅ **SIGNAL IS NON-SILENT** (Speech data detected)")
                
                # Save WAV
                wav_path = r"D:\JCAB\research\validation\mic_capture_test.wav"
                sf.write(wav_path, data, samplerate, subtype='PCM_16', format='WAV')
                st.success(f"Saved decoded WAV to: `{wav_path}`")
                
        except Exception as e:
            st.error(f"Failed to decode WAV: {e}")
            
    else:
        st.error("Received 0 bytes from the recorder component.")
