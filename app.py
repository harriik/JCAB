"""
JCAB
Java Conversational Assessment and Personalized Feedback System

Research Title: 
"An Adaptive Personalized Feedback Framework for Enhancing 
Multi-Agent Large Language Model-Based Conversational Assessment in Education"

This is a preliminary academic research prototype using a synthetic development dataset.
"""

from __future__ import annotations

import logging
from datetime import datetime
import pandas as pd
import numpy as np
import streamlit as st

import soundfile as sf
import io
from pathlib import Path
from streamlit_mic_recorder import mic_recorder

# Local modules
from config.settings import (
    APP_NAME, APP_VERSION, EMBEDDING_MODEL_NAME, EXPECTED_EMBEDDING_DIMENSION,
    DOMAINS, DATASET_STATUS, RESEARCH_STATUS, TARGET_POPULATION, USE_LLM_FEEDBACK
)
from modules.preprocessing import build_conversation
from modules.embeddings import load_embedding_model, generate_embedding
from modules.prediction import (
    decode_learner_level,
    load_ml_models,
    predict_level,
    predict_score,
)
from modules.feedback import get_feedback
from modules.assessment import get_random_questions

# ============================================================
# 1. LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(APP_NAME)

# ============================================================
# 2. STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Adaptive Assessment",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional educational dashboard CSS supporting Light, Dark, and System themes
st.markdown(
    """
    <style>
    /* University Educational Dashboard Design System - Adaptive Theme Tokens */
    :root {
        color-scheme: light dark;
        --edu-bg: light-dark(#F3F6F8, #0F172A);
        --edu-surface: light-dark(#FFFFFF, #1E293B);
        --edu-surface-muted: light-dark(#F8FAFC, #162032);
        --edu-text: light-dark(#1F2937, #F8FAFC);
        --edu-text-secondary: light-dark(#64748B, #94A3B8);
        --edu-text-muted: light-dark(#94A3B8, #64748B);
        --edu-accent: light-dark(#168675, #2DD4BF);
        --edu-accent-hover: light-dark(#126E60, #14B8A6);
        --edu-accent-surface: light-dark(#E8F5F1, rgba(45, 212, 191, 0.12));
        --edu-accent-border: light-dark(#A7F3D0, rgba(45, 212, 191, 0.28));
        --edu-border: light-dark(#E2E8F0, #334155);
        --edu-divider: light-dark(#CBD5E1, #475569);
        --edu-card-shadow: light-dark(0 1px 3px rgba(15, 23, 42, 0.04), 0 2px 6px rgba(0, 0, 0, 0.3));

        --edu-bullet-green-bg: light-dark(#F0FDF4, rgba(22, 163, 74, 0.15));
        --edu-bullet-green-border: light-dark(#BBF7D0, rgba(22, 163, 74, 0.35));
        --edu-bullet-green-text: light-dark(#166534, #86EFAC);

        --edu-bullet-amber-bg: light-dark(#FFFBEB, rgba(217, 119, 6, 0.15));
        --edu-bullet-amber-border: light-dark(#FDE68A, rgba(217, 119, 6, 0.35));
        --edu-bullet-amber-text: light-dark(#92400E, #FDE68A);
    }

    @media (prefers-color-scheme: dark) {
        :root {
            --edu-bg: #0F172A;
            --edu-surface: #1E293B;
            --edu-surface-muted: #162032;
            --edu-text: #F8FAFC;
            --edu-text-secondary: #94A3B8;
            --edu-text-muted: #64748B;
            --edu-accent: #2DD4BF;
            --edu-accent-hover: #14B8A6;
            --edu-accent-surface: rgba(45, 212, 191, 0.12);
            --edu-accent-border: rgba(45, 212, 191, 0.28);
            --edu-border: #334155;
            --edu-divider: #475569;
            --edu-card-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);

            --edu-bullet-green-bg: rgba(22, 163, 74, 0.15);
            --edu-bullet-green-border: rgba(22, 163, 74, 0.35);
            --edu-bullet-green-text: #86EFAC;

            --edu-bullet-amber-bg: rgba(217, 119, 6, 0.15);
            --edu-bullet-amber-border: rgba(217, 119, 6, 0.35);
            --edu-bullet-amber-text: #FDE68A;
        }
    }
    
    .stApp {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
    }
    
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1040px !important;
    }

    /* Sidebar polish - preserve native theme background with adaptive border */
    section[data-testid="stSidebar"] {
        border-right: 1px solid var(--edu-border) !important;
    }

    /* Primary and Secondary Buttons */
    div.stButton > button[kind="primary"] {
        background-color: var(--edu-accent) !important;
        color: light-dark(#FFFFFF, #0F172A) !important;
        border: none !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        padding: 0.55rem 1.25rem !important;
        transition: all 0.15s ease-in-out !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: var(--edu-accent-hover) !important;
        box-shadow: 0 2px 5px rgba(22, 134, 117, 0.25) !important;
    }
    div.stButton > button[kind="secondary"] {
        background-color: var(--edu-surface) !important;
        color: var(--edu-text) !important;
        border: 1px solid var(--edu-border) !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
        padding: 0.55rem 1.25rem !important;
        transition: all 0.15s ease-in-out !important;
    }
    div.stButton > button[kind="secondary"]:hover {
        background-color: var(--edu-surface-muted) !important;
        border-color: var(--edu-accent) !important;
    }

    /* Cards & Containers */
    .edu-card {
        background-color: var(--edu-surface);
        border: 1px solid var(--edu-border);
        border-radius: 8px;
        padding: 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: var(--edu-card-shadow);
        color: var(--edu-text);
    }
    
    .edu-hero {
        background-color: var(--edu-surface);
        border: 1px solid var(--edu-border);
        border-radius: 8px;
        padding: 2.25rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: var(--edu-card-shadow);
        color: var(--edu-text);
    }

    /* Step Indicator */
    .step-tracker {
        display: flex;
        align-items: center;
        justify-content: space-between;
        background-color: var(--edu-surface);
        border: 1px solid var(--edu-border);
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        margin-bottom: 1.75rem;
        font-size: 0.875rem;
        box-shadow: var(--edu-card-shadow);
    }
    .step-item {
        color: var(--edu-text-secondary);
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.4rem;
    }
    .step-item.active {
        color: var(--edu-accent);
        font-weight: 700;
    }
    .step-item.completed {
        color: var(--edu-text);
        font-weight: 600;
    }
    .step-divider {
        color: var(--edu-divider);
        font-weight: 300;
    }

    /* Question & Response Boxes */
    .edu-q-box {
        background-color: var(--edu-surface);
        border: 1px solid var(--edu-border);
        border-left: 4px solid var(--edu-accent);
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.25rem;
        box-shadow: var(--edu-card-shadow);
    }
    .edu-q-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--edu-accent);
        margin-bottom: 0.4rem;
    }
    .edu-q-text {
        font-size: 1.1rem;
        font-weight: 600;
        color: var(--edu-text);
        line-height: 1.55;
    }
    .edu-resp-box {
        background-color: var(--edu-surface-muted);
        border: 1px solid var(--edu-border);
        border-radius: 8px;
        padding: 1rem 1.25rem;
        margin-bottom: 1.25rem;
    }
    .edu-resp-label {
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--edu-text-secondary);
        margin-bottom: 0.35rem;
    }
    .edu-resp-text {
        font-size: 1rem;
        color: var(--edu-text);
        line-height: 1.5;
    }

    /* Educational Result Metric Cards */
    .edu-metric-card {
        background-color: var(--edu-surface);
        border: 1px solid var(--edu-border);
        border-radius: 8px;
        padding: 1.25rem;
        text-align: center;
        box-shadow: var(--edu-card-shadow);
    }
    .edu-metric-val {
        font-size: 1.85rem;
        font-weight: 700;
        color: var(--edu-accent);
        line-height: 1.2;
    }
    .edu-metric-lbl {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: var(--edu-text-secondary);
        margin-top: 0.35rem;
    }

    .edu-section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: var(--edu-text);
        margin-top: 0;
        margin-bottom: 0.75rem;
    }

    .edu-bullet-green {
        background-color: var(--edu-bullet-green-bg);
        border: 1px solid var(--edu-bullet-green-border);
        border-radius: 6px;
        padding: 0.65rem 0.9rem;
        margin-bottom: 0.5rem;
        color: var(--edu-bullet-green-text);
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .edu-bullet-amber {
        background-color: var(--edu-bullet-amber-bg);
        border: 1px solid var(--edu-bullet-amber-border);
        border-radius: 6px;
        padding: 0.65rem 0.9rem;
        margin-bottom: 0.5rem;
        color: var(--edu-bullet-amber-text);
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .edu-bullet-action {
        background-color: var(--edu-surface);
        border: 1px solid var(--edu-border);
        border-left: 3px solid var(--edu-accent);
        border-radius: 6px;
        padding: 0.65rem 0.9rem;
        margin-bottom: 0.5rem;
        color: var(--edu-text);
        font-size: 0.95rem;
        line-height: 1.5;
    }

    .edu-feedback-box {
        background-color: var(--edu-surface-muted);
        border: 1px solid var(--edu-border);
        border-left: 4px solid #3B82F6;
        border-radius: 6px;
        padding: 1.15rem 1.35rem;
        font-size: 1rem;
        line-height: 1.6;
        color: var(--edu-text);
        margin-bottom: 1.25rem;
    }

    .edu-concept-box {
        background-color: var(--edu-surface);
        border: 1px solid var(--edu-border);
        border-left: 4px solid var(--edu-accent);
        border-radius: 6px;
        padding: 1.15rem 1.35rem;
        font-size: 1rem;
        line-height: 1.6;
        color: var(--edu-text);
        margin-bottom: 1.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# 3. SESSION STATE ROUTER
# ============================================================

DEFAULT_SESSION_STATE = {
    "current_page": "home", # home, setup, chat, processing, results, feedback, dashboard, report
    "assessment_result": None,
    "assessment_history": [],
    "current_student": "",
    "current_id": "",
    "current_level": "Beginner",
    "selected_domain": "Java",
    "assessment_questions": [],
    "current_q_index": 0,
    "current_stage": "initial", # initial or followup
    "initial_response": "",
    "followup_response": "",
    "initial_resp_text": "",
    "followup_resp_text": "",
    "last_audio": None,
    "last_audio_2": None,
    "initial_input_mode": "text",
    "followup_input_mode": "text",
    "demo_mode": False,
    "consent_given": False,
    "survey_completed": False
}

for key, default_value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value

def navigate_to(page_name: str):
    st.session_state.current_page = page_name

def reset_assessment():
    st.session_state.assessment_questions = []
    st.session_state.current_q_index = 0
    st.session_state.current_stage = "initial"
    st.session_state.initial_response = ""
    st.session_state.followup_response = ""
    st.session_state.initial_resp_text = ""
    st.session_state.followup_resp_text = ""
    st.session_state.last_audio = None
    st.session_state.last_audio_2 = None
    st.session_state.initial_input_mode = "text"
    st.session_state.followup_input_mode = "text"
    st.session_state.assessment_result = None
    navigate_to("setup")

# ============================================================
# 4. INITIALIZE MODELS
# ============================================================

models_ready = False
model_error = None

try:
    with st.spinner("Loading AI Engine..."):
        embedding_model = load_embedding_model()
        score_model, level_model, encoder_model = load_ml_models()
    models_ready = True
except Exception as exc:
    model_error = str(exc)
    embedding_model = None
    score_model = None
    level_model = None
    encoder_model = None
    logger.exception("Model initialization failed")

# ============================================================
# 5. SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("<div style='font-size: 1.25rem; font-weight: 700; color: var(--edu-text); margin-bottom: 0.15rem; line-height: 1.2;'>Adaptive Assessment</div>", unsafe_allow_html=True)
    st.markdown("<div style='font-size: 0.8rem; color: var(--edu-text-secondary); margin-bottom: 1.25rem;'>Personalized Educational Assessment</div>", unsafe_allow_html=True)
    
    st.markdown("<div style='font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--edu-text-muted); margin-bottom: 0.5rem;'>Navigation</div>", unsafe_allow_html=True)
    
    cur_p = st.session_state.current_page
    if st.button("🏠 Home", use_container_width=True, type="primary" if cur_p == "home" else "secondary"):
        navigate_to("home")
        st.rerun()
    if st.button("📝 Start Assessment", use_container_width=True, type="primary" if cur_p in ["setup", "chat"] else "secondary"):
        reset_assessment()
        st.rerun()
    if st.button("📊 Student Dashboard", use_container_width=True, type="primary" if cur_p in ["dashboard", "results", "evaluation"] else "secondary"):
        navigate_to("dashboard")
        st.rerun()
        
    st.divider()
    
    st.markdown("<div style='font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--edu-text-muted); margin-bottom: 0.5rem;'>Settings</div>", unsafe_allow_html=True)
    st.session_state.demo_mode = st.toggle("Demo Mode (Safe Fallbacks)", value=st.session_state.demo_mode)
    if st.session_state.demo_mode:
        st.caption("⚡ Local deterministic fallback active")
    else:
        st.caption("🤖 Multi-agent assessment pipeline active")
        
    st.divider()
    st.caption("Adaptive Assessment v0.2.0 • Academic Decision Support")


if not models_ready and not st.session_state.demo_mode:
    st.error("### Application cannot start")
    st.write("One or more preliminary trained models could not be loaded. Please activate DEMO MODE to bypass.")
    if model_error:
        st.error(f"Model loading error:\n\n{model_error}")
    st.stop()


# ============================================================
# HELPER: SPEECH TO TEXT
# ============================================================

@st.cache_resource
def get_whisper_model():
    from faster_whisper import WhisperModel
    logger.info("Loading Whisper model (tiny.en) to CPU...")
    # tiny.en is small enough for CPU and very fast
    return WhisperModel("tiny.en", device="cpu", compute_type="int8")

def transcribe_voice(audio_bytes: bytes) -> tuple[bool, str, str, str, float, float, dict]:
    """
    Attempts to transcribe raw WAV audio bytes using faster-whisper.
    Returns: (success, transcript, error_type, error_message, duration, latency, diagnostics)
    """
    import time
    import numpy as np
    import soundfile as sf
    import io
    import os
    
    start_time = time.time()
    diagnostics = {}
    
    try:
        raw_size = len(audio_bytes)
        logger.info(f"Raw audio byte size: {raw_size}")
        
        if raw_size == 0:
            return False, "", "NO_AUDIO", "No microphone audio was captured.", 0.0, 0.0, diagnostics
            
        hex_preview = audio_bytes[:32].hex()
        logger.info(f"Raw audio first 32 bytes (hex): {hex_preview}")
        
        # Parse WAV bytes into numpy array using soundfile
        try:
            with sf.SoundFile(io.BytesIO(audio_bytes)) as sfile:
                logger.info(f"WAV Info - channels: {sfile.channels}, samplerate: {sfile.samplerate}, subtype: {sfile.subtype}, format: {sfile.format}, frames: {sfile.frames}")
                data = sfile.read(dtype='float32')
                samplerate = sfile.samplerate
        except Exception as e:
            logger.error(f"Failed to decode WAV audio: {e}")
            return False, "", "DECODE_ERROR", "The recorded audio could not be decoded.", 0.0, 0.0, diagnostics
            
        # If stereo, convert to mono
        if len(data.shape) > 1:
            data = data.mean(axis=1)
            
        if samplerate != 16000:
            logger.info(f"Resampling from {samplerate} to 16000")
            import scipy.signal
            number_of_samples = round(len(data) * float(16000) / samplerate)
            data = scipy.signal.resample(data, number_of_samples)
            samplerate = 16000
            
        sample_count = len(data)
        duration = sample_count / 16000.0
        
        # Calculate Diagnostics
        min_amp = float(np.min(data))
        max_amp = float(np.max(data))
        mean_abs_amp = float(np.mean(np.abs(data)))
        rms = float(np.sqrt(np.mean(data**2)))
        nonzero_pct = float(np.mean(data != 0)) * 100
        
        # RMS per 1-second segment
        per_second_rms = []
        for i in range(int(np.ceil(duration))):
            start_idx = i * 16000
            end_idx = min((i + 1) * 16000, sample_count)
            seg = data[start_idx:end_idx]
            if len(seg) > 0:
                seg_rms = float(np.sqrt(np.mean(seg**2)))
                per_second_rms.append(round(seg_rms, 4))
                
        diagnostics = {
            "sample_rate": samplerate,
            "duration": round(duration, 2),
            "samples": sample_count,
            "dtype": str(data.dtype),
            "min": round(min_amp, 4),
            "max": round(max_amp, 4),
            "mean_abs": round(mean_abs_amp, 4),
            "rms": round(rms, 4),
            "nonzero_pct": round(nonzero_pct, 2),
            "per_second_rms": per_second_rms
        }
        
        logger.info("VOICE DEBUG\n" + "\n".join([f"{k}={v}" for k, v in diagnostics.items()]))
        
        # Save exact decoded audio used by Whisper
        try:
            debug_dir = os.path.join("research", "validation")
            os.makedirs(debug_dir, exist_ok=True)
            debug_path = os.path.join(debug_dir, "debug_voice.wav")
            sf.write(debug_path, data, samplerate, subtype='PCM_16', format='WAV')
        except Exception as e:
            logger.warning(f"Failed to save debug_voice.wav: {e}")
            
        # CRITICAL DECISION LOGIC A vs B
        if rms < 0.0005:
            logger.error("CASE A: Audio is effectively silent. Aborting transcription.")
            return False, "", "SILENT_AUDIO", "No usable microphone signal detected. Check microphone permissions and selected input device.", duration, 0.0, diagnostics
            
        logger.info("CASE B: Audio has healthy amplitude. Proceeding to Whisper.")
        
        if duration < 0.5:
            return False, "", "TOO_SHORT", "The recording was too short. Please speak for at least a moment.", duration, 0.0, diagnostics
            
        logger.info("Transcribing speech with faster-whisper...")
        
        model = get_whisper_model()
        
        # Whisper settings explicitly for normal English speech
        segments, info = model.transcribe(
            data, 
            language="en", 
            task="transcribe",
            temperature=0,
            beam_size=5, 
            vad_filter=False,
            condition_on_previous_text=False
        )
        
        transcript_parts = []
        for i, segment in enumerate(segments):
            logger.info(f"SEGMENT {i+1}: [{segment.start:.2f} - {segment.end:.2f}] '{segment.text}' (avg_logprob: {segment.avg_logprob:.2f}, no_speech_prob: {segment.no_speech_prob:.2f})")
            transcript_parts.append(segment.text)
            
        transcript = "".join(transcript_parts).strip()
        latency = time.time() - start_time
        diagnostics["transcript"] = transcript
        logger.info(f"Transcription successful in {latency:.2f}s. Text: {transcript}")
        
        if not transcript:
            return False, "", "NO_SPEECH", "No speech was detected.", duration, latency, diagnostics
            
        # Suspicious tiny transcript check for non-silent audio
        words = transcript.split()
        if duration > 4.0 and len(words) <= 2:
            return False, "", "SUSPICIOUS_TRANSCRIPT", f"Audio was captured but transcription failed to detect a full sentence (detected: '{transcript}'). Please type your answer or retry.", duration, latency, diagnostics
            
        return True, transcript, "", "", duration, latency, diagnostics
            
    except Exception as e:
        logger.error(f"Transcription Failed: {e}", exc_info=True)
        return False, "", "STT_FAILED", "Speech transcription failed. You can type your answer instead.", 0.0, 0.0, diagnostics


# ============================================================
# PAGE: HOME
# ============================================================

def page_home():
    st.markdown(
        """
        <div class="edu-hero">
            <h1 style="color: var(--edu-text); margin: 0 0 0.35rem 0; font-size: 2.25rem; font-weight: 700; letter-spacing: -0.02em;">Adaptive Assessment</h1>
            <p style="color: var(--edu-accent); margin: 0 0 1rem 0; font-size: 1.15rem; font-weight: 600; font-style: italic;">Understand your answers. Improve your learning.</p>
            <p style="color: var(--edu-text-secondary); margin: 0; font-size: 1.05rem; line-height: 1.6; max-width: 700px;">
                Explore conceptual questions, receive feedback on your responses, and discover what to learn next.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, _ = st.columns([1.5, 1.5, 2.5])
    with col1:
        if st.button("Start Assessment", type="primary", use_container_width=True):
            reset_assessment()
            st.rerun()
    with col2:
        if st.button("Student Dashboard", use_container_width=True):
            navigate_to("dashboard")
            st.rerun()
            
    st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            """
            <div class="edu-card" style="height: 100%;">
                <div style="font-size: 0.85rem; font-weight: 700; letter-spacing: 0.05em; color: var(--edu-accent); text-transform: uppercase; margin-bottom: 0.5rem;">Assess</div>
                <div style="font-size: 0.95rem; color: var(--edu-text-secondary); line-height: 1.55;">
                    Answer conceptual questions through a structured assessment.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        st.markdown(
            """
            <div class="edu-card" style="height: 100%;">
                <div style="font-size: 0.85rem; font-weight: 700; letter-spacing: 0.05em; color: var(--edu-accent); text-transform: uppercase; margin-bottom: 0.5rem;">Understand</div>
                <div style="font-size: 0.95rem; color: var(--edu-text-secondary); line-height: 1.55;">
                    Review your strengths and identify concepts that need improvement.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        st.markdown(
            """
            <div class="edu-card" style="height: 100%;">
                <div style="font-size: 0.85rem; font-weight: 700; letter-spacing: 0.05em; color: var(--edu-accent); text-transform: uppercase; margin-bottom: 0.5rem;">Improve</div>
                <div style="font-size: 0.95rem; color: var(--edu-text-secondary); line-height: 1.55;">
                    Explore short concept explanations and actionable learning recommendations.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# PAGE: SETUP (STUDENT PROFILE)
# ============================================================

def page_setup():
    st.markdown(
        """
        <div class="step-tracker">
            <span class="step-item active">● Step 1: Setup</span>
            <span class="step-divider">→</span>
            <span class="step-item">○ Step 2: Initial Answer</span>
            <span class="step-divider">→</span>
            <span class="step-item">○ Step 3: Follow-up</span>
            <span class="step-divider">→</span>
            <span class="step-item">○ Step 4: Results</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="font-size: 1.6rem; font-weight: 700; color: var(--edu-text); margin: 0 0 0.3rem 0;">Student Profile & Assessment Setup</h2>
            <p style="color: var(--edu-text-secondary); font-size: 0.95rem; margin: 0;">Configure your participant details and select your assessment preferences.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    with st.container():
        st.markdown(
            """
            <div class="edu-card" style="margin-bottom: 1rem; background-color: var(--edu-surface-muted);">
                <div style="font-size: 0.85rem; font-weight: 700; color: var(--edu-text); margin-bottom: 0.35rem;">Academic Research & Participation Notice</div>
                <div style="font-size: 0.9rem; color: var(--edu-text-secondary); line-height: 1.55;">
                    Your participation is voluntary. Assessment responses and feedback are collected to evaluate and refine personalized educational conversational systems. All participant identifiers may be anonymous.
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        consent = st.checkbox("I agree to participate in this academic research assessment", value=st.session_state.consent_given)
        st.session_state.consent_given = consent
    
    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
    
    with st.form("setup_form"):
        st.markdown("<div style='font-weight: 700; color: var(--edu-text); margin-bottom: 0.75rem; font-size: 1.05rem;'>1. Participant Information</div>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        student_name = c1.text_input("Full Name or Identifier", value=st.session_state.current_student, placeholder="e.g. Student A / Anonymous")
        student_id = c2.text_input("Participant ID", value=st.session_state.current_id, placeholder="e.g. MCA-2026-01")
        level = c3.selectbox("Self-Assessed Knowledge Level", ["Beginner", "Intermediate", "Advanced"], index=["Beginner", "Intermediate", "Advanced"].index(st.session_state.current_level) if st.session_state.current_level in ["Beginner", "Intermediate", "Advanced"] else 0)
        
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight: 700; color: var(--edu-text); margin-bottom: 0.75rem; font-size: 1.05rem;'>2. Assessment Domain & Scope</div>", unsafe_allow_html=True)
        
        domain_opts = list(DOMAINS.keys())
        selected_dom = st.selectbox(
            "Select Subject Domain", 
            domain_opts, 
            index=0,
            help="Java is currently the active assessment domain. Other domains are scheduled for future academic releases."
        )
        
        dom_status = DOMAINS[selected_dom]["status"]
        dom_desc = DOMAINS[selected_dom]["description"]
        
        if dom_status == "available":
            st.markdown(f"<div style='background-color: var(--edu-accent-surface); border: 1px solid var(--edu-accent-border); border-radius: 6px; padding: 0.75rem 1rem; color: var(--edu-accent); font-size: 0.9rem; margin-bottom: 1rem;'><b>Active Domain:</b> {selected_dom} — {dom_desc}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='background-color: var(--edu-bullet-amber-bg); border: 1px solid var(--edu-bullet-amber-border); border-radius: 6px; padding: 0.75rem 1rem; color: var(--edu-bullet-amber-text); font-size: 0.9rem; margin-bottom: 1rem;'><b>Upcoming Domain:</b> {selected_dom} (Planned in future releases) — {dom_desc}</div>", unsafe_allow_html=True)
            
        st.markdown("<div style='height: 0.5rem;'></div>", unsafe_allow_html=True)
        st.markdown("<div style='font-weight: 700; color: var(--edu-text); margin-bottom: 0.5rem; font-size: 1.05rem;'>3. Assessment Length</div>", unsafe_allow_html=True)
        num_questions = st.slider("Number of Questions to Complete", min_value=1, max_value=5, value=2, help="Choose between 1 and 5 questions for this assessment session.")
            
        st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Start Assessment", type="primary", use_container_width=True)
        
        if submitted:
            if not st.session_state.consent_given:
                st.error("Please indicate agreement with the participation notice to begin the assessment.")
            elif not student_name.strip() or not student_id.strip():
                st.error("Please provide a name/identifier and Participant ID.")
            elif dom_status != "available":
                st.error("Please select an active domain (Java). Other domains are planned for future implementation.")
            else:
                st.session_state.current_student = student_name.strip()
                st.session_state.current_id = student_id.strip()
                st.session_state.current_level = level
                st.session_state.selected_domain = selected_dom
                
                questions = get_random_questions(num_questions)
                if not questions:
                    questions = [{"topic": "Java Core", "question": "Explain OOP principles in Java.", "followup": "Explain Inheritance and method overriding."}]
                    
                st.session_state.assessment_questions = questions
                st.session_state.current_q_index = 0
                st.session_state.current_stage = "initial"
                st.session_state.initial_resp_text = ""
                st.session_state.followup_resp_text = ""
                st.session_state.last_audio = None
                st.session_state.last_audio_2 = None
                st.session_state.initial_input_mode = "text"
                st.session_state.followup_input_mode = "text"
                
                navigate_to("chat")
                st.rerun()


# ============================================================
# PAGE: CONVERSATIONAL CHAT
# ============================================================

def page_chat():
    q_idx = st.session_state.current_q_index
    total_q = len(st.session_state.assessment_questions)
    
    if q_idx >= total_q or total_q == 0:
        st.warning("No questions found. Returning to setup.")
        reset_assessment()
        st.rerun()
        return
        
    current_q_data = st.session_state.assessment_questions[q_idx]
    is_initial = (st.session_state.current_stage == "initial")
    
    # 1. Step Indicator
    if is_initial:
        st.markdown(
            """
            <div class="step-tracker">
                <span class="step-item completed">✓ Step 1: Setup</span>
                <span class="step-divider">→</span>
                <span class="step-item active">● Step 2: Initial Answer</span>
                <span class="step-divider">→</span>
                <span class="step-item">○ Step 3: Follow-up</span>
                <span class="step-divider">→</span>
                <span class="step-item">○ Step 4: Results</span>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="step-tracker">
                <span class="step-item completed">✓ Step 1: Setup</span>
                <span class="step-divider">→</span>
                <span class="step-item completed">✓ Step 2: Initial Answer</span>
                <span class="step-divider">→</span>
                <span class="step-item active">● Step 3: Follow-up</span>
                <span class="step-divider">→</span>
                <span class="step-item">○ Step 4: Results</span>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # 2. Progress Bar
    step_ratio = (q_idx + (0.5 if not is_initial else 0.1)) / max(1, total_q)
    st.progress(min(1.0, max(0.01, step_ratio)), text=f"Question {q_idx + 1} of {total_q} • Topic: {current_q_data['topic']}")
    st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
    
    if is_initial:
        # Question Card
        st.markdown(
            f"""
            <div class="edu-q-box">
                <div class="edu-q-label">Assessment Question • Topic: {current_q_data['topic']}</div>
                <div class="edu-q-text">{current_q_data['question']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Voice input controls
        vcol1, vcol2 = st.columns([1.6, 3])
        with vcol1:
            audio_val = mic_recorder(
                start_prompt="🎙️ Speak your answer", 
                stop_prompt="🔴 Listening... Speak your answer now", 
                key="mic_initial", 
                format="wav"
            )
        with vcol2:
            st.caption("Click to record your voice answer or type directly in the response box below.")
            
        if audio_val is not None:
            audio_bytes = audio_val['bytes']
            audio_hash = hash(audio_bytes)
            if st.session_state.get('last_audio_hash_initial') != audio_hash:
                with st.spinner("Processing audio... Transcribing response..."):
                    success, transcript, err_type, err_msg, dur, lat, diagnostics = transcribe_voice(audio_bytes)
                    st.session_state.voice_diagnostics = diagnostics
                    if success and transcript:
                        st.session_state.initial_resp_text = transcript
                        st.session_state.initial_input_mode = "voice"
                        st.session_state.transcription_notice = "Review your response and correct any transcription errors before continuing."
                    else:
                        st.session_state.transcription_notice = "Microphone input could not be captured. You can type your response in the box below."
                    st.session_state.last_audio_hash_initial = audio_hash
                    st.rerun()
                    
        if st.session_state.get('transcription_notice'):
            if "could not be captured" in st.session_state.transcription_notice:
                st.warning(st.session_state.transcription_notice)
            else:
                st.info(st.session_state.transcription_notice)
                
        st.text_area(
            "Your Response", 
            key="initial_resp_text", 
            height=160, 
            placeholder="Type your explanation here or record using the microphone..."
        )
        
        if st.session_state.get('voice_diagnostics') and st.session_state.get('demo_mode'):
            with st.expander("Developer Voice Diagnostics"):
                st.json(st.session_state.voice_diagnostics)
                
        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
        bcol1, bcol2 = st.columns([1.5, 3])
        with bcol1:
            if st.button("Submit Answer", type="primary", use_container_width=True):
                if len(st.session_state.initial_resp_text.strip()) < 5:
                    st.warning("Please provide a conceptual response before continuing.")
                else:
                    st.session_state.initial_response = st.session_state.initial_resp_text.strip()
                    st.session_state.current_stage = "followup"
                    st.session_state.voice_diagnostics = None
                    st.session_state.transcription_notice = None
                    st.rerun()
                    
    else:
        # Show previous exchange in compact educational card
        st.markdown(
            f"""
            <div class="edu-resp-box">
                <div class="edu-resp-label">Initial Question & Your Response</div>
                <div style="font-weight: 600; color: var(--edu-text); margin-bottom: 0.35rem;">{current_q_data['question']}</div>
                <div class="edu-resp-text">{st.session_state.initial_response}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        # Follow-up Question Card
        st.markdown(
            f"""
            <div class="edu-q-box">
                <div class="edu-q-label">Follow-up Question • Topic: {current_q_data['topic']}</div>
                <div class="edu-q-text">{current_q_data['followup']}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        vcol1, vcol2 = st.columns([1.6, 3])
        with vcol1:
            audio_val_2 = mic_recorder(
                start_prompt="🎙️ Speak your answer", 
                stop_prompt="🔴 Listening... Speak your answer now", 
                key="mic_followup", 
                format="wav"
            )
        with vcol2:
            st.caption("Click to record your voice answer or type directly in the response box below.")
            
        if audio_val_2 is not None:
            audio_bytes = audio_val_2['bytes']
            audio_hash = hash(audio_bytes)
            if st.session_state.get('last_audio_hash_followup') != audio_hash:
                with st.spinner("Processing audio... Transcribing response..."):
                    success, transcript_2, err_type, err_msg_2, dur, lat, diagnostics = transcribe_voice(audio_bytes)
                    st.session_state.voice_diagnostics = diagnostics
                    if success and transcript_2:
                        st.session_state.followup_resp_text = transcript_2
                        st.session_state.followup_input_mode = "voice"
                        st.session_state.transcription_notice = "Review your response and correct any transcription errors before continuing."
                    else:
                        st.session_state.transcription_notice = "Microphone input could not be captured. You can type your response in the box below."
                    st.session_state.last_audio_hash_followup = audio_hash
                    st.rerun()
                    
        if st.session_state.get('transcription_notice'):
            if "could not be captured" in st.session_state.transcription_notice:
                st.warning(st.session_state.transcription_notice)
            else:
                st.info(st.session_state.transcription_notice)
                
        st.text_area(
            "Your Follow-up Response", 
            key="followup_resp_text", 
            height=160, 
            placeholder="Type your follow-up explanation here or record using the microphone..."
        )
        
        if st.session_state.get('voice_diagnostics') and st.session_state.get('demo_mode'):
            with st.expander("Developer Voice Diagnostics"):
                st.json(st.session_state.voice_diagnostics)
                
        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
        bcol1, bcol2 = st.columns([1.5, 3])
        with bcol1:
            if st.button("Complete Assessment", type="primary", use_container_width=True):
                if len(st.session_state.followup_resp_text.strip()) < 5:
                    st.warning("Please provide a conceptual response before continuing.")
                else:
                    st.session_state.followup_response = st.session_state.followup_resp_text.strip()
                    st.session_state.voice_diagnostics = None
                    st.session_state.transcription_notice = None
                    navigate_to("processing")
                    st.rerun()

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    if st.button("Cancel & Return to Setup"):
        st.session_state.voice_diagnostics = None
        st.session_state.transcription_notice = None
        reset_assessment()
        st.rerun()


def normalize_assessment_result(
    feedback_data: dict,
    base_info: dict
) -> dict:
    """
    Normalizes the assessment result into a consistent internal structure.
    Guarantees authoritative score, learner_level, feedback content, and mode are present
    and type-safe, preventing KeyErrors.
    """
    res = dict(base_info)
    res.update(feedback_data)

    # 1. Authoritative Score & Learner Level
    auth_score = feedback_data.get("score")
    auth_level = feedback_data.get("learner_level")

    if auth_score is None:
        try:
            from modules.feedback import generate_authoritative_assessment
            auth_calc = generate_authoritative_assessment(
                base_info.get("topic", "Java"),
                base_info.get("question", ""),
                base_info.get("initial_response", ""),
                base_info.get("followup_question", ""),
                base_info.get("followup_response", "")
            )
            auth_score = auth_calc.get("score", 5.0)
            auth_level = auth_calc.get("learner_level", "Partial")
        except Exception:
            auth_score = 5.0
            auth_level = "Partial"

    if not auth_level:
        if auth_score >= 8.5:
            auth_level = "Excellent"
        elif auth_score >= 7.0:
            auth_level = "Good"
        elif auth_score >= 5.0:
            auth_level = "Partial"
        elif auth_score >= 2.5:
            auth_level = "Weak"
        else:
            auth_level = "Incorrect"

    res["score"] = round(float(auth_score), 2)
    res["learner_level"] = str(auth_level)

    # 2. Experimental ML Outputs (kept separate, never overwrites authoritative score)
    res["experimental_ml_score"] = float(feedback_data.get("experimental_ml_score", base_info.get("experimental_ml_score", 0.0)))
    res["experimental_ml_learner_level"] = str(feedback_data.get("experimental_ml_learner_level", base_info.get("experimental_ml_learner_level", "Unknown")))

    # 3. Content fields
    summary = feedback_data.get("overall_summary") or feedback_data.get("feedback") or "Assessment completed successfully."
    feedback_text = feedback_data.get("feedback") or feedback_data.get("personalized_feedback") or summary
    res["overall_summary"] = str(summary).strip()
    res["feedback"] = str(feedback_text).strip()
    res["personalized_feedback"] = res["feedback"]

    # 4. List fields (strengths, improvements, misconceptions, actions)
    res["strengths"] = list(feedback_data.get("strengths", []))[:2]
    res["improvements"] = list(feedback_data.get("improvements", feedback_data.get("areas_to_improve", [])))[:2]
    res["areas_to_improve"] = res["improvements"]
    res["misconceptions"] = list(feedback_data.get("misconceptions", []))[:2]

    rec_actions = list(feedback_data.get("recommended_actions", feedback_data.get("practice_suggestions", [])))[:2]
    if not rec_actions and feedback_data.get("recommendation"):
        rec_actions = [str(feedback_data.get("recommendation"))]
    res["recommended_actions"] = rec_actions[:2]
    res["practice_suggestions"] = rec_actions[:2]
    res["recommendation"] = feedback_data.get("recommendation") or (rec_actions[0] if rec_actions else "")
    res["next_step"] = feedback_data.get("next_step", "")

    # 5. Feedback Mode reporting
    mode = feedback_data.get("mode") or feedback_data.get("feedback_mode") or "Demo Mode — Local Fallback"
    res["mode"] = str(mode)
    res["feedback_mode"] = str(mode)

    # 6. Concept Explanation (Topic Reference Knowledge)
    concept_exp = feedback_data.get("concept_explanation") or base_info.get("concept_explanation", "")
    if not concept_exp:
        try:
            import json
            from pathlib import Path
            rubric_path = Path("data/java_fallback_rubrics.json")
            if rubric_path.exists():
                with open(rubric_path, "r", encoding="utf-8") as f:
                    rubrics = json.load(f)
                topic = base_info.get("topic", "")
                concept_exp = rubrics.get(topic, {}).get("reference_explanation", "")
        except Exception:
            pass
    if not concept_exp:
        concept_exp = f"In Java, {base_info.get('topic', 'this topic')} establishes core language conventions, structural mechanics, and design guidelines."
    res["concept_explanation"] = str(concept_exp).strip()

    return res


# ============================================================
# PAGE: PROCESSING
# ============================================================

def page_processing():
    st.subheader("Analyzing Response...")
    
    q_idx = st.session_state.current_q_index
    current_q_data = st.session_state.assessment_questions[q_idx]
    
    progress_placeholder = st.empty()
    
    def update_progress(msg: str):
        progress_placeholder.info(msg)
        
    try:
        # Determine execution mode
        is_demo = bool(st.session_state.get("demo_mode", False))
        should_use_llm = (not is_demo) and bool(USE_LLM_FEEDBACK)
        
        # 1. Experimental ML Pipeline (Sentence-BERT + XGBoost)
        predicted_score = 0.0
        predicted_level = "Models Offline"
        if models_ready and embedding_model is not None:
            try:
                update_progress("Generating text embeddings...")
                conversation_text = build_conversation(st.session_state.initial_response, st.session_state.followup_response)
                embedding = generate_embedding(conversation_text, embedding_model)
                if score_model is not None and level_model is not None and encoder_model is not None:
                    update_progress("Computing experimental ML metrics...")
                    predicted_score = predict_score(embedding, score_model)
                    predicted_level_encoded = predict_level(embedding, level_model)
                    predicted_level = decode_learner_level(predicted_level_encoded, encoder_model)
            except Exception as ml_err:
                logger.warning(f"Experimental ML pipeline warning (non-fatal): {ml_err}")
                
        # 2. Authoritative Assessment & Feedback Generation
        if should_use_llm:
            update_progress("Running Multi-Agent Feedback Pipeline...")
        else:
            update_progress("Generating assessment feedback (Local Fallback)...")
            
        feedback_data = get_feedback(
            current_q_data['question'],
            st.session_state.initial_response,
            current_q_data['followup'],
            st.session_state.followup_response,
            predicted_score,
            predicted_level,
            current_q_data['topic'],
            use_llm=should_use_llm,
            demo_mode=is_demo,
            progress_callback=update_progress
        )
            
        progress_placeholder.empty()
        
        base_metadata = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "student_name": st.session_state.current_student,
            "student_id": st.session_state.current_id,
            "domain": "Java",
            "topic": current_q_data['topic'],
            "question": current_q_data['question'],
            "initial_response": st.session_state.initial_response,
            "followup_question": current_q_data['followup'],
            "followup_response": st.session_state.followup_response,
            "initial_input_mode": st.session_state.initial_input_mode,
            "followup_input_mode": st.session_state.followup_input_mode,
            "experimental_ml_score": predicted_score,
            "experimental_ml_learner_level": predicted_level
        }
        
        # Merge and normalize into consistent structure
        result = normalize_assessment_result(feedback_data, base_metadata)
        
        st.session_state.assessment_result = result
        st.session_state.assessment_history.append(result)
        
        # Write to JSONL log for research
        import json
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        with open(log_dir / "assessment_research_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(result) + "\n")

        # Save actual validation data
        val_dir = Path("research/validation")
        val_dir.mkdir(parents=True, exist_ok=True)
        raw_csv = val_dir / "student_validation_raw.csv"
        
        df_new = pd.DataFrame([{
            "participant_id": result.get("student_id", ""),
            "timestamp": result.get("timestamp", ""),
            "domain": result.get("domain", "Java"),
            "question_id": current_q_data.get('id', 'Q_UNKNOWN'),
            "topic": result.get("topic", ""),
            "assessment_question": result.get("question", ""),
            "initial_response": result.get("initial_response", ""),
            "followup_question": result.get("followup_question", ""),
            "followup_response": result.get("followup_response", ""),
            "response_input_mode": "voice" if result.get("initial_input_mode") == "voice" or result.get("followup_input_mode") == "voice" else "text",
            "predicted_score": result.get("score", 0.0),
            "learner_level": result.get("learner_level", "Unknown"),
            "feedback_mode": result.get("mode", "UNKNOWN"),
            "generated_feedback": result.get("overall_summary", result.get("feedback", "")),
            "recommendation": result.get("recommendation", "")
        }])
        
        if not raw_csv.exists():
            df_new.to_csv(raw_csv, index=False)
        else:
            df_new.to_csv(raw_csv, mode='a', header=False, index=False)
        
        # Check if more questions
        total_q = len(st.session_state.assessment_questions)
        if q_idx + 1 < total_q:
            st.session_state.current_q_index += 1
            st.session_state.current_stage = "initial"
            st.session_state.initial_resp_text = ""
            st.session_state.followup_resp_text = ""
            st.session_state.last_audio = None
            st.session_state.last_audio_2 = None
            st.session_state.initial_input_mode = "text"
            st.session_state.followup_input_mode = "text"
            navigate_to("chat")
        else:
            navigate_to("results")
            
        st.rerun()
        
    except Exception as exc:
        logger.exception("Assessment processing encountered an unexpected issue")
        st.error("### Assessment Processing Encountered An Error")
        st.warning(f"Details: {exc}")
        st.info("You may retry this response or return to the setup page.")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Retry Response", type="primary", use_container_width=True):
                navigate_to("chat")
                st.rerun()
        with col2:
            if st.button("Return to Setup", use_container_width=True):
                reset_assessment()
                st.rerun()



# ============================================================
# PAGE: RESULTS (COMBINED ASSESSMENT & FEEDBACK)
# ============================================================

def page_results():
    result = st.session_state.assessment_result
    if not result:
        st.warning("No assessment results available. Returning to home.")
        navigate_to("home")
        st.rerun()
        return
        
    st.markdown(
        """
        <div class="step-tracker">
            <span class="step-item completed">✓ Step 1: Setup</span>
            <span class="step-divider">→</span>
            <span class="step-item completed">✓ Step 2: Initial Answer</span>
            <span class="step-divider">→</span>
            <span class="step-item completed">✓ Step 3: Follow-up</span>
            <span class="step-divider">→</span>
            <span class="step-item active">● Step 4: Results & Report</span>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="font-size: 1.75rem; font-weight: 700; color: var(--edu-text); margin: 0 0 0.35rem 0;">Assessment Report</h2>
            <p style="color: var(--edu-text-secondary); font-size: 0.95rem; margin: 0;">Comprehensive evaluation of conceptual understanding and educational recommendations.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # Top compact metric row (Assessment Score, Learner Level, Topic)
    m1, m2, m3 = st.columns(3)
    score_val = float(result.get('score', 0.0))
    lvl_val = str(result.get('learner_level', 'Partial'))
    topic_val = str(result.get('topic', 'Java'))
    
    lvl_color = "var(--edu-accent)" if lvl_val in ["Excellent", "Good"] else ("#F59E0B" if lvl_val == "Partial" else "#F87171")
    
    with m1:
        st.markdown(
            f"""
            <div class="edu-metric-card">
                <div class="edu-metric-val">{score_val:.2f} <span style="font-size: 1.05rem; font-weight: 500; color: var(--edu-text-secondary);">/ 10</span></div>
                <div class="edu-metric-lbl">Assessment Score</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m2:
        st.markdown(
            f"""
            <div class="edu-metric-card">
                <div class="edu-metric-val" style="color: {lvl_color};">{lvl_val}</div>
                <div class="edu-metric-lbl">Learner Level</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with m3:
        st.markdown(
            f"""
            <div class="edu-metric-card">
                <div class="edu-metric-val" style="font-size: 1.35rem; color: var(--edu-text); padding-top: 0.35rem;">{topic_val}</div>
                <div class="edu-metric-lbl">Assessed Topic</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("<div style='height: 1.25rem;'></div>", unsafe_allow_html=True)
    
    # A. Assessment Summary (One concise sentence describing overall result)
    summary_text = result.get('overall_summary', '').strip()
    if summary_text:
        st.markdown(
            f"""
            <div class="edu-card" style="margin-bottom: 1.25rem;">
                <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; color: var(--edu-text-secondary); margin-bottom: 0.35rem;">Assessment Summary</div>
                <div style="font-size: 1.05rem; font-weight: 600; color: var(--edu-text); line-height: 1.5;">{summary_text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    # B. Strengths & C. Areas for Improvement (Side-by-side)
    col_str, col_imp = st.columns(2)
    strengths = [s for s in result.get('strengths', []) if s][:2]
    improvements = [i for i in result.get('improvements', result.get('areas_to_improve', [])) if i][:2]
    misconceptions = [m for m in result.get('misconceptions', []) if m][:2]

    with col_str:
        st.markdown("<div class='edu-section-title'>✅ Strengths</div>", unsafe_allow_html=True)
        if strengths:
            for s in strengths:
                st.markdown(f"<div class='edu-bullet-green'>• {s}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='edu-bullet-green'>• Foundational response provided; review principles to demonstrate conceptual depth.</div>", unsafe_allow_html=True)

    with col_imp:
        st.markdown("<div class='edu-section-title'>📉 Areas for Improvement</div>", unsafe_allow_html=True)
        if improvements:
            for i in improvements:
                st.markdown(f"<div class='edu-bullet-amber'>• {i}</div>", unsafe_allow_html=True)
        else:
            st.markdown("<div class='edu-bullet-amber'>• Core concepts addressed accurately with no critical gaps identified.</div>", unsafe_allow_html=True)
            
        if misconceptions:
            st.markdown("<div style='font-size: 0.85rem; font-weight: 700; color: #991B1B; margin-top: 0.75rem; margin-bottom: 0.35rem;'>⚠️ Misconceptions Detected</div>", unsafe_allow_html=True)
            for m in misconceptions:
                st.markdown(f"<div class='edu-bullet-amber' style='border-color: #FECACA; background-color: #FEF2F2; color: #991B1B;'>• {m}</div>", unsafe_allow_html=True)

    # D. Feedback on Your Answer (How student performed on actual question)
    st.markdown("<div class='edu-section-title' style='margin-top: 1.5rem;'>💬 Feedback on Your Answer</div>", unsafe_allow_html=True)
    feedback_text = result.get('feedback', result.get('personalized_feedback', '')).strip()
    if not feedback_text:
        feedback_text = summary_text
    st.markdown(f"<div class='edu-feedback-box'>{feedback_text}</div>", unsafe_allow_html=True)

    # E. Short Concept Explanation (Separate educational section explaining topic itself)
    st.markdown(f"<div class='edu-section-title' style='margin-top: 1.5rem;'>📖 Short Concept Explanation: {topic_val}</div>", unsafe_allow_html=True)
    concept_explanation = result.get('concept_explanation', '').strip()
    if not concept_explanation:
        concept_explanation = f"In Java, {topic_val} forms a key structural mechanism governing object interaction, code organization, and execution rules."
    st.markdown(
        f"""
        <div class="edu-concept-box">
            <div style="font-weight: 700; color: #168675; margin-bottom: 0.4rem; font-size: 0.95rem;">Key Principles & Mechanism</div>
            <div style="color: #334155; line-height: 1.6;">{concept_explanation}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # F. Recommended Next Steps (Actionable checklist)
    st.markdown("<div class='edu-section-title' style='margin-top: 1.5rem;'>🎯 Recommended Next Steps</div>", unsafe_allow_html=True)
    actions = result.get('recommended_actions', [])
    if not actions:
        actions = result.get('practice_suggestions', [])
    if not actions:
        rec = result.get('recommendation', '')
        ns = result.get('next_step', '')
        actions = [a for a in [rec, ns] if a][:2]
    actions = [a for a in actions if a][:2]
    if not actions:
        actions = ["Review foundational Java documentation for this topic.", "Practice writing a minimal code example to solidify syntax and rules."]

    for act in actions:
        st.markdown(f"<div class='edu-bullet-action'>👉 {act}</div>", unsafe_allow_html=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

    # Developer Diagnostic Note (Only in Demo Mode, no ML comparison models)
    if st.session_state.get("demo_mode"):
        with st.expander("System Configuration (Demo Mode Enabled)"):
            mode = result.get('mode', 'Demo Mode — Local Fallback')
            st.caption(f"Active Assessment Engine: {mode}")

    # Navigation Actions
    col1, col2 = st.columns([1.5, 1.5])
    with col1:
        if st.button("Continue to Feedback Survey", type="primary", use_container_width=True):
            st.session_state.survey_completed = False
            navigate_to("evaluation")
            st.rerun()
    with col2:
        if st.button("View Student Dashboard", use_container_width=True):
            navigate_to("dashboard")
            st.rerun()

def page_feedback():
    # Deprecated page - route to results
    navigate_to("results")
    st.rerun()


# ============================================================
# PAGE: EVALUATION (STUDENT SURVEY)
# ============================================================

# ============================================================
# PAGE: EVALUATION (STUDENT SURVEY)
# ============================================================

def page_evaluation():
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="font-size: 1.6rem; font-weight: 700; color: #1F2937; margin: 0 0 0.3rem 0;">Student Assessment Feedback</h2>
            <p style="color: #64748B; font-size: 0.95rem; margin: 0;">Help us evaluate the clarity, relevance, and helpfulness of the assessment feedback you received.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown(
        """
        <div class="edu-card" style="margin-bottom: 1.25rem; background-color: #F8FAFC;">
            <div style="font-size: 0.85rem; font-weight: 600; color: #475569;">
                Rate each statement on a scale from 1 (Strongly Disagree) to 5 (Strongly Agree).
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    with st.form("evaluation_form"):
        r1 = st.slider("1. The feedback was relevant to my answer.", 1, 5, 4)
        r2 = st.slider("2. The feedback helped me understand my areas for improvement.", 1, 5, 4)
        r3 = st.slider("3. The feedback was specific to what I wrote.", 1, 5, 4)
        r4 = st.slider("4. The recommended next steps were useful and actionable.", 1, 5, 4)
        r5 = st.slider("5. The conversational assessment interface was easy to use.", 1, 5, 4)
        
        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
        useful_text = st.text_area("What was most helpful about the feedback?", placeholder="Share what worked well for you...")
        improve_text = st.text_area("What should be improved in the assessment or feedback?", placeholder="Suggestions for improvement...")
        
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("Submit Survey & View Dashboard", type="primary", use_container_width=True)
        
        if submitted:
            val_dir = Path("research/validation")
            val_dir.mkdir(parents=True, exist_ok=True)
            survey_csv = val_dir / "student_feedback_ratings.csv"
            
            result = st.session_state.assessment_result
            
            df_survey = pd.DataFrame([{
                "participant_id": st.session_state.current_id,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "topic": result["topic"] if result else "",
                "q1_relevant": r1,
                "q2_helpful": r2,
                "q3_specific": r3,
                "q4_recommendation": r4,
                "q5_usability": r5,
                "most_useful": useful_text,
                "needs_improvement": improve_text
            }])
            
            if not survey_csv.exists():
                df_survey.to_csv(survey_csv, index=False)
            else:
                df_survey.to_csv(survey_csv, mode='a', header=False, index=False)
            
            st.session_state.survey_completed = True
            navigate_to("dashboard")
            st.rerun()


# ============================================================
# PAGE: DASHBOARD & REPORT
# ============================================================

def page_dashboard():
    st.markdown(
        """
        <div style="margin-bottom: 1.5rem;">
            <h2 style="font-size: 1.75rem; font-weight: 700; color: #1F2937; margin: 0 0 0.35rem 0;">Student Learning Dashboard</h2>
            <p style="color: #64748B; font-size: 0.95rem; margin: 0;">Overview of your completed assessments, progress trends, and learning recommendations.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    history = st.session_state.assessment_history
    
    if not history:
        st.markdown(
            """
            <div class="edu-card" style="text-align: center; padding: 3rem 2rem;">
                <div style="font-size: 2.25rem; margin-bottom: 0.75rem;">📚</div>
                <h3 style="color: #1F2937; margin-bottom: 0.5rem; font-size: 1.25rem;">No Assessments Completed Yet</h3>
                <p style="color: #64748B; max-width: 480px; margin: 0 auto 1.5rem auto; line-height: 1.55;">
                    Complete your first conceptual assessment to unlock your personalized progress tracker, score breakdown, and study recommendations.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )
        c1, c2, c3 = st.columns([1.5, 1, 1.5])
        with c2:
            if st.button("Start Assessment", type="primary", use_container_width=True):
                reset_assessment()
                st.rerun()
        return
        
    df = pd.DataFrame(history)
    
    # 1. Summary Metrics
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            f"""
            <div class="edu-metric-card">
                <div class="edu-metric-val">{len(df)}</div>
                <div class="edu-metric-lbl">Total Completed</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c2:
        avg_score = float(df['score'].mean()) if 'score' in df.columns else 0.0
        st.markdown(
            f"""
            <div class="edu-metric-card">
                <div class="edu-metric-val">{avg_score:.2f}</div>
                <div class="edu-metric-lbl">Average Score</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c3:
        last_lvl = str(df.iloc[-1]['learner_level']) if 'learner_level' in df.columns else "N/A"
        lvl_color = "#168675" if last_lvl in ["Excellent", "Good"] else ("#B45309" if last_lvl == "Partial" else "#991B1B")
        st.markdown(
            f"""
            <div class="edu-metric-card">
                <div class="edu-metric-val" style="color: {lvl_color}; font-size: 1.6rem;">{last_lvl}</div>
                <div class="edu-metric-lbl">Current Level</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with c4:
        last_top = str(df.iloc[-1]['topic']) if 'topic' in df.columns else "N/A"
        st.markdown(
            f"""
            <div class="edu-metric-card">
                <div class="edu-metric-val" style="font-size: 1.25rem; color: #1F2937; padding-top: 0.35rem;">{last_top}</div>
                <div class="edu-metric-lbl">Latest Topic</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    
    # 2. Performance by Topic & Assessment History
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='edu-section-title'>📈 Performance by Topic</div>", unsafe_allow_html=True)
        topic_scores = df.groupby("topic")["score"].mean()
        st.bar_chart(topic_scores)
        
    with col2:
        st.markdown("<div class='edu-section-title'>📋 Recent Assessments</div>", unsafe_allow_html=True)
        display_cols = [c for c in ["timestamp", "topic", "score", "learner_level"] if c in df.columns]
        st.dataframe(df[display_cols], use_container_width=True, hide_index=True)

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    
    # 3. Repeated Focus Areas & Recommendations
    all_improvements = []
    for item in history:
        for imp in item.get('improvements', []):
            if imp and imp not in all_improvements:
                all_improvements.append(imp)
                
    if all_improvements:
        st.markdown("<div class='edu-section-title'>🎯 Focus Areas Identified Across Sessions</div>", unsafe_allow_html=True)
        for imp in all_improvements[:4]:
            st.markdown(f"<div class='edu-bullet-amber'>• {imp}</div>", unsafe_allow_html=True)
        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

    # 4. Detailed Session Review
    st.markdown("<div class='edu-section-title'>📖 Detailed Session Review</div>", unsafe_allow_html=True)
    for i, row in df.iterrows():
        with st.expander(f"{row.get('timestamp', '')} • Topic: {row.get('topic', '')} (Score: {row.get('score', 0):.2f}/10 - {row.get('learner_level', '')})"):
            st.markdown(f"**Question:** {row.get('question', '')}")
            st.markdown(f"> *Your Response:* {row.get('initial_response', '')}")
            if row.get('followup_question'):
                st.markdown(f"**Follow-up Question:** {row.get('followup_question', '')}")
                st.markdown(f"> *Your Follow-up Response:* {row.get('followup_response', '')}")
            st.markdown("---")
            st.markdown(f"**Performance Feedback:** {row.get('feedback', row.get('overall_summary', ''))}")
            if row.get('concept_explanation'):
                st.markdown(f"**Topic Concept Reference:** {row.get('concept_explanation', '')}")
            if row.get('recommendation'):
                st.markdown(f"**Recommended Action:** {row.get('recommendation', '')}")
            
    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    
    # 5. Clean Student Report Download (Excludes all internal ML model comparisons)
    student_report_cols = [
        "timestamp", "student_name", "topic", "score", "learner_level", 
        "question", "initial_response", "followup_question", "followup_response", 
        "overall_summary", "feedback", "concept_explanation", "recommendation"
    ]
    export_df = df[[c for c in student_report_cols if c in df.columns]].copy()
    csv_data = export_df.to_csv(index=False).encode("utf-8")
    safe_name = st.session_state.current_student.strip().replace(" ", "_") or "Student"
    
    st.download_button(
        label="📥 Download Personal Assessment Report (CSV)",
        data=csv_data,
        file_name=f"Adaptive_Assessment_Report_{safe_name}.csv",
        mime="text/csv",
        type="primary"
    )

    st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)
    
    # 6. Academic Research Data Export (Preserved in collapsed expander for researchers)
    with st.expander("Academic Research Data Exports (Authorized Access)"):
        st.caption("Authorized research datasets containing participant validation records and feedback ratings.")
        col1, col2 = st.columns(2)
        
        val_raw = Path("research/validation/student_validation_raw.csv")
        if val_raw.exists():
            with open(val_raw, "rb") as f:
                col1.download_button("📥 Validation Data (Raw CSV)", data=f, file_name="student_validation_raw.csv", mime="text/csv")
                
        val_survey = Path("research/validation/student_feedback_ratings.csv")
        if val_survey.exists():
            with open(val_survey, "rb") as f:
                col2.download_button("📥 Feedback Ratings (CSV)", data=f, file_name="student_feedback_ratings.csv", mime="text/csv")

# ============================================================
# MAIN ROUTER
# ============================================================

pages = {
    "home": page_home,
    "setup": page_setup,
    "chat": page_chat,
    "processing": page_processing,
    "results": page_results,
    "feedback": page_feedback,
    "evaluation": page_evaluation,
    "dashboard": page_dashboard,
    "report": page_dashboard # Report and Dashboard merged for cleaner UX
}

current_page_func = pages.get(st.session_state.current_page, page_home)
current_page_func()