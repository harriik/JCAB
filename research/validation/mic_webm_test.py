import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import io
import av

st.set_page_config(page_title="WebM Mic Test", page_icon="🎤")
st.title("Final WebM Capture & Hardware Diagnostic")

st.info("⚠️ Ensure you click **Allow** when Chrome asks for microphone permissions, and check that the correct physical microphone is selected in Chrome settings.")

# 1. LIVE BROWSER HARDWARE TEST
st.header("1. Live Browser Hardware Diagnostic")
st.markdown("Use this to test if Chrome is *actually* receiving microphone audio levels before recording.")
components.html(
    """
    <div style="font-family: sans-serif; padding: 15px; border: 1px solid #ccc; border-radius: 5px; background: #fafafa;">
        <h4 style="margin-top:0;">Live Microphone RMS Analyzer</h4>
        <button id="startBtn" style="padding: 8px 16px; background: #ff4b4b; color: white; border: none; border-radius: 4px; cursor: pointer;">Start Live Diagnostic</button>
        <p><strong>Status:</strong> <span id="status">Waiting to start...</span></p>
        <p><strong>Selected Device:</strong> <span id="deviceInfo">None</span></p>
        <p><strong>RMS Level:</strong> <span id="rmsValue" style="font-weight:bold; font-size: 1.2em;">0.0000</span></p>
        <div style="width: 100%; background: #e0e0e0; height: 30px; border-radius: 4px; overflow: hidden;">
            <div id="meter" style="width: 0%; background: #4caf50; height: 100%; transition: width 0.1s;"></div>
        </div>
    </div>
    <script>
        document.getElementById('startBtn').addEventListener('click', async () => {
            const statusEl = document.getElementById('status');
            try {
                statusEl.innerText = "Requesting microphone...";
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                statusEl.innerText = "Active. Speak into your microphone.";
                
                const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                const source = audioCtx.createMediaStreamSource(stream);
                const analyser = audioCtx.createAnalyser();
                analyser.fftSize = 2048;
                source.connect(analyser);
                
                const dataArray = new Float32Array(analyser.fftSize);
                
                const track = stream.getAudioTracks()[0];
                document.getElementById('deviceInfo').innerText = track.label || track.id;
                
                function update() {
                    analyser.getFloatTimeDomainData(dataArray);
                    let sumSquares = 0.0;
                    for (let i = 0; i < dataArray.length; i++) {
                        sumSquares += dataArray[i] * dataArray[i];
                    }
                    let rms = Math.sqrt(sumSquares / dataArray.length);
                    document.getElementById('rmsValue').innerText = rms.toFixed(4);
                    // Scale meter so 0.1 is 100% for visibility
                    document.getElementById('meter').style.width = Math.min(100, rms * 1000) + '%';
                    requestAnimationFrame(update);
                }
                update();
            } catch (err) {
                statusEl.innerText = "ERROR: " + err.message;
                document.getElementById('deviceInfo').innerText = "N/A";
            }
        });
    </script>
    """,
    height=250
)

# 2. WEBM RECORDER CAPTURE
st.header("2. WebM Component Capture")
from streamlit_mic_recorder import mic_recorder

audio_val = mic_recorder(
    start_prompt="🎤 Start WebM Recording", 
    stop_prompt="⏹️ Stop Recording", 
    key="mic_webm", 
    format="webm"
)

if audio_val is not None:
    audio_bytes = audio_val['bytes']
    raw_size = len(audio_bytes)
    st.write(f"**Browser Recorder Status:** Success")
    st.write(f"**Raw Byte Size:** {raw_size} bytes")
    
    if raw_size > 0:
        raw_path = r"D:\JCAB\research\validation\mic_webm_capture.webm"
        with open(raw_path, "wb") as f:
            f.write(audio_bytes)
        st.success(f"Saved raw WebM to: `{raw_path}`")
        
        st.markdown("### Playback Test")
        st.markdown("If you hear nothing here, your microphone captured silence.")
        st.audio(audio_bytes, format="audio/webm")
        
        # Decode using PyAV
        st.markdown("### PyAV Decoder Test")
        try:
            frames = []
            with av.open(io.BytesIO(audio_bytes), mode="r") as container:
                stream = container.streams.audio[0]
                # Extract to s16 mono 16kHz for absolute stability
                resampler = av.AudioResampler(format="s16", layout="mono", rate=16000)
                
                for frame in container.decode(stream):
                    r_frames = resampler.resample(frame)
                    for r_frame in r_frames:
                        frames.append(r_frame.to_ndarray().flatten())
                for r_frame in resampler.resample(None):
                    frames.append(r_frame.to_ndarray().flatten())
            
            if not frames:
                st.error("PyAV returned no frames.")
            else:
                data_s16 = np.concatenate(frames)
                data = data_s16.astype(np.float32) / 32768.0
                
                duration = len(data) / 16000.0
                min_amp = float(np.min(data))
                max_amp = float(np.max(data))
                mean_abs = float(np.mean(np.abs(data)))
                rms = float(np.sqrt(np.mean(data**2)))
                nonzero_pct = float(np.mean(data != 0)) * 100
                
                st.write("**Decoded Audio Diagnostics:**")
                st.json({
                    "duration_seconds": round(duration, 2),
                    "min_amplitude": round(min_amp, 6),
                    "max_amplitude": round(max_amp, 6),
                    "mean_absolute": round(mean_abs, 6),
                    "RMS": round(rms, 6),
                    "nonzero_percentage": round(nonzero_pct, 2)
                })
                
                if rms < 0.0005:
                    st.error("🚨 **DECODED SIGNAL IS SILENT** (RMS effectively zero)")
                else:
                    st.success("✅ **DECODED SIGNAL IS NON-SILENT** (Speech data detected)")
                    
        except Exception as e:
            st.error(f"Failed to decode WebM with PyAV: {e}")
