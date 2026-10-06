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

# Professional academic aesthetic CSS
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1100px;
    }
    .header-box {
        padding: 2.5rem;
        background-color: var(--secondary-background-color);
        border-radius: 6px;
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 2.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    .header-title {
        font-size: 2.75rem;
        font-weight: 800;
        color: var(--text-color);
        margin: 0;
        line-height: 1.2;
    }
    .header-subtitle {
        font-size: 1.25rem;
        font-weight: 500;
        color: var(--text-color);
        opacity: 0.9;
        margin-top: 0.75rem;
        margin-bottom: 1rem;
    }
    .header-support {
        font-size: 1.1rem;
        color: var(--text-color);
        opacity: 0.7;
        margin-top: 0.5rem;
    }
    .chat-msg {
        padding: 1.25rem;
        border-radius: 6px;
        margin-bottom: 1.25rem;
        font-size: 1.05rem;
        line-height: 1.6;
        border: 1px solid rgba(128, 128, 128, 0.2);
        color: var(--text-color);
    }
    .chat-ai {
        background-color: var(--secondary-background-color);
    }
    .chat-student {
        background-color: rgba(128, 128, 128, 0.05);
    }
    .chat-label {
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
        color: var(--text-color);
        opacity: 0.7;
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
    st.title("Adaptive Assessment")
    st.caption("Personalized Educational Assessment")
    
    st.divider()
    
    st.markdown("**Navigation**")
    if st.button("🏠 Home", use_container_width=True):
        navigate_to("home")
    if st.button("📊 Dashboard", use_container_width=True):
        navigate_to("dashboard")
    if st.button("📝 Start Assessment", use_container_width=True, type="primary"):
        reset_assessment()
        
    st.divider()
    
    st.markdown("**Research Configuration**")
    st.session_state.demo_mode = st.toggle("Demo Mode (Safe Fallbacks)", value=st.session_state.demo_mode)
    if st.session_state.demo_mode:
        st.warning("DEMO MODE ACTIVE")
        
    st.info(f"**Dataset:** {DATASET_STATUS}\n\n**Status:** {RESEARCH_STATUS}")
    
    if models_ready:
        st.success("✅ Models Online")
    else:
        st.error("❌ Models Offline")


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
        <div class="header-box">
            <h1 class="header-title">Adaptive Assessment</h1>
            <div class="header-subtitle">
                Personalized AI-Based Educational Assessment
            </div>
            <div class="header-support">
                Assess conceptual understanding through conversational responses and provide personalized feedback and learning recommendations.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    col1, col2, col3 = st.columns([2, 2, 4])
    with col1:
        if st.button("Start Assessment", type="primary", use_container_width=True):
            reset_assessment()
    with col2:
        if st.button("View Student Dashboard", use_container_width=True):
            navigate_to("dashboard")
            
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**ASSESS**<br>Conversational assessment of conceptual understanding", unsafe_allow_html=True)
    with c2:
        st.markdown("**ANALYZE**<br>Semantic and rubric-based analysis of student responses", unsafe_allow_html=True)
    with c3:
        st.markdown("**PERSONALIZE**<br>Individualized feedback and learning recommendations", unsafe_allow_html=True)


# ============================================================
# PAGE: SETUP (STUDENT PROFILE)
# ============================================================

def page_setup():
    st.subheader("1. Research Consent & Student Profile")
    
    st.markdown("### Consent / Participation Notice")
    st.info("Your participation is voluntary. Responses are collected for academic research and evaluation of the prototype.")
    consent = st.checkbox("I Agree and Continue", value=st.session_state.consent_given)
    st.session_state.consent_given = consent
    
    st.divider()
    
    with st.form("setup_form"):
        st.markdown("**Participant Details**")
        c1, c2, c3 = st.columns(3)
        student_name = c1.text_input("Name (or 'Anonymous')", value=st.session_state.current_student)
        student_id = c2.text_input("Participant ID (can be anonymous)", value=st.session_state.current_id)
        level = c3.selectbox("Current Learner Level", ["Beginner", "Intermediate", "Advanced"], index=["Beginner", "Intermediate", "Advanced"].index(st.session_state.current_level) if st.session_state.current_level in ["Beginner", "Intermediate", "Advanced"] else 0)
        
        st.markdown("**Select Assessment Domain**")
        
        domain_opts = list(DOMAINS.keys())
        selected_dom = st.selectbox("Domain", domain_opts, index=0)
        
        dom_status = DOMAINS[selected_dom]["status"]
        dom_desc = DOMAINS[selected_dom]["description"]
        
        if dom_status == "available":
            st.success(f"**{selected_dom} (Available):** {dom_desc}")
        else:
            st.warning(f"**{selected_dom} (Coming Soon):** {dom_desc}")
            
        st.markdown("**Assessment Length**")
        num_questions = st.slider("Number of Questions", min_value=1, max_value=5, value=2)
            
        submitted = st.form_submit_button("Start Assessment", type="primary")
        
        if submitted:
            if not st.session_state.consent_given:
                st.error("You must agree to the participation notice to continue.")
            elif not student_name.strip() or not student_id.strip():
                st.error("Please enter a Participant ID and Name (or use an anonymous identifier).")
            elif dom_status != "available":
                st.error("This domain is planned for future implementation. Please select an available domain (Java).")
            else:
                st.session_state.current_student = student_name
                st.session_state.current_id = student_id
                st.session_state.current_level = level
                st.session_state.selected_domain = selected_dom
                
                # Fetch questions directly from the expanded question bank
                questions = get_random_questions(num_questions)
                if not questions:
                    # Failsafe fallback
                    questions = [{"topic": "Java Core", "question": "Explain OOP.", "followup": "Explain Inheritance."}]
                    
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
    st.subheader(f"Conversational Assessment: {st.session_state.selected_domain}")
    
    q_idx = st.session_state.current_q_index
    total_q = len(st.session_state.assessment_questions)
    
    if q_idx >= total_q or total_q == 0:
        st.warning("No questions found. Returning to setup.")
        reset_assessment()
        return
        
    current_q_data = st.session_state.assessment_questions[q_idx]
    
    # Render Progress
    st.progress(max(0.01, q_idx / total_q), text=f"Question {q_idx + 1} of {total_q} | Topic: {current_q_data['topic']}")
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Render AI Initial Question
    st.markdown("<div class='chat-label'>Question</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='chat-msg chat-ai'>🤖 <b>AI Assessor:</b><br/>{current_q_data['question']}</div>", unsafe_allow_html=True)
    
    if st.session_state.current_stage == "initial":
        # Audio Input
        audio_val = mic_recorder(
            start_prompt="🎙️ Speak", 
            stop_prompt="🔴 Listening...", 
            key="mic_initial", 
            format="wav"
        )
        if audio_val is not None:
            audio_bytes = audio_val['bytes']
            audio_hash = hash(audio_bytes)
            if st.session_state.get('last_audio_hash_initial') != audio_hash:
                with st.spinner("Processing audio... Transcribing..."):
                    success, transcript, err_type, err_msg, dur, lat, diagnostics = transcribe_voice(audio_bytes)
                    st.session_state.voice_diagnostics = diagnostics
                    if success:
                        st.session_state.initial_resp_text = transcript
                        st.session_state.initial_input_mode = "voice"
                    else:
                        st.error("Microphone input could not be captured. You can type your answer instead.")
                    st.session_state.last_audio_hash_initial = audio_hash
                    st.rerun()
                    
        # Text Input (bound to session state so STT updates it automatically)
        st.text_area("Your Response", key="initial_resp_text", height=150, placeholder="Type your response here or use the microphone...")
        
        if st.session_state.get('voice_diagnostics') and st.session_state.get('demo_mode'):
            with st.expander("Developer Voice Diagnostics"):
                st.json(st.session_state.voice_diagnostics)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Submit Response", type="primary"):
            if len(st.session_state.initial_resp_text.strip()) < 5:
                st.warning("Please provide a valid response.")
            else:
                st.session_state.initial_response = st.session_state.initial_resp_text
                st.session_state.current_stage = "followup"
                st.session_state.voice_diagnostics = None # Reset
                st.rerun()
                
    elif st.session_state.current_stage == "followup":
        # Show previous interaction
        st.markdown("<div class='chat-label'>Your Response</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-msg chat-student'>👤 <b>You:</b><br/>{st.session_state.initial_response}</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='chat-label'>Follow-up Question</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-msg chat-ai'>🤖 <b>AI Follow-up:</b><br/>{current_q_data['followup']}</div>", unsafe_allow_html=True)
        
        audio_val_2 = mic_recorder(
            start_prompt="🎙️ Speak", 
            stop_prompt="🔴 Listening...", 
            key="mic_followup", 
            format="wav"
        )
        if audio_val_2 is not None:
            audio_bytes = audio_val_2['bytes']
            audio_hash = hash(audio_bytes)
            if st.session_state.get('last_audio_hash_followup') != audio_hash:
                with st.spinner("Processing audio... Transcribing..."):
                    success, transcript_2, err_type, err_msg_2, dur, lat, diagnostics = transcribe_voice(audio_bytes)
                    st.session_state.voice_diagnostics = diagnostics
                    if success:
                        st.session_state.followup_resp_text = transcript_2
                        st.session_state.followup_input_mode = "voice"
                    else:
                        st.error("Microphone input could not be captured. You can type your answer instead.")
                    st.session_state.last_audio_hash_followup = audio_hash
                    st.rerun()
                    
        st.text_area("Your Follow-up Response", key="followup_resp_text", height=150, placeholder="Type your response here or use the microphone...")
        
        if st.session_state.get('voice_diagnostics') and st.session_state.get('demo_mode'):
            with st.expander("Developer Voice Diagnostics"):
                st.json(st.session_state.voice_diagnostics)
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Complete Assessment", type="primary"):
            if len(st.session_state.followup_resp_text.strip()) < 5:
                st.warning("Please provide a valid response.")
            else:
                st.session_state.followup_response = st.session_state.followup_resp_text
                st.session_state.voice_diagnostics = None # Reset
                navigate_to("processing")
                st.rerun()

    st.divider()
    if st.button("End Assessment Early"):
        st.session_state.voice_diagnostics = None
        reset_assessment()


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
        if st.session_state.demo_mode or not models_ready:
            # Safe Fallback for Demo Mode
            predicted_score = float(np.random.uniform(5.0, 9.5))
            predicted_level = "Intermediate"
            predicted_level_encoded = 1
            feedback_data = {
                "strengths": ["Demonstrated basic understanding of the topic.", "Identified key concepts correctly."],
                "improvements": ["Could provide more concrete examples.", "Expand on the theoretical background."],
                "feedback": f"Your explanation of {current_q_data['topic']} was good, but adding more detailed examples would make it much stronger.",
                "recommendation": f"Review {current_q_data['topic']} with practical coding exercises.",
                "mode": "demo"
            }
        else:
            # True ML Pipeline
            update_progress("Generating text embeddings...")
            conversation_text = build_conversation(st.session_state.initial_response, st.session_state.followup_response)
            embedding = generate_embedding(conversation_text, embedding_model)
            
            update_progress("Predicting teacher score and learner level...")
            predicted_score = predict_score(embedding, score_model)
            predicted_level_encoded = predict_level(embedding, level_model)
            predicted_level = decode_learner_level(predicted_level_encoded, encoder_model)
            
            feedback_data = get_feedback(
                current_q_data['question'],
                st.session_state.initial_response,
                current_q_data['followup'],
                st.session_state.followup_response,
                predicted_score,
                predicted_level,
                current_q_data['topic'],
                use_llm=USE_LLM_FEEDBACK,
                progress_callback=update_progress
            )
            
        progress_placeholder.empty()
        
        result = {
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
            "followup_input_mode": st.session_state.followup_input_mode
        }
        
        # Merge all feedback keys safely (this includes authoritative score, learner_level, and experimental ML values)
        result.update(feedback_data)
        
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
            "participant_id": result["student_id"],
            "timestamp": result["timestamp"],
            "domain": result["domain"],
            "question_id": current_q_data.get('id', 'Q_UNKNOWN'),
            "topic": result["topic"],
            "assessment_question": result["question"],
            "initial_response": result["initial_response"],
            "followup_question": result["followup_question"],
            "followup_response": result["followup_response"],
            "response_input_mode": "voice" if result.get("initial_input_mode") == "voice" or result.get("followup_input_mode") == "voice" else "text",
            "predicted_score": result["score"],
            "learner_level": result["learner_level"],
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
        logger.exception("Assessment execution failed")
        st.error("An error occurred during semantic processing.")
        st.exception(exc)
        if st.button("Return to Setup"):
            reset_assessment()


# ============================================================
# PAGE: RESULTS (COMBINED ASSESSMENT & FEEDBACK)
# ============================================================

def page_results():
    result = st.session_state.assessment_result
    
    st.markdown("<h2 style='text-align: center; color: var(--text-color); margin-bottom: 2rem;'>Assessment Report</h2>", unsafe_allow_html=True)
    
    # OVERVIEW METRICS
    st.markdown("<div style='padding: 1.5rem; background-color: var(--secondary-background-color); border: 1px solid rgba(128,128,128,0.2); border-radius: 8px; margin-bottom: 2rem;'>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Score", f"{result['score']:.2f} / 10")
    c2.metric("Learner Level", result["learner_level"])
    c3.metric("Topic", result["topic"])
    st.markdown("</div>", unsafe_allow_html=True)
    
    # PERSONALIZED FEEDBACK (Visually prominent)
    st.markdown("### Personalized Feedback", unsafe_allow_html=True)
    
    mode = result.get('mode', 'Development Fallback (Rule-Based)')
    if mode == "Multi-Agent LLM":
        if result.get('overall_summary'):
            st.markdown(f"<p style='font-size: 1.15rem; color: var(--text-color); font-weight: 500;'>{result['overall_summary']}</p>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div style='padding: 1.5rem; background-color: rgba(34, 197, 94, 0.1); border-left: 5px solid #22c55e; border-radius: 4px; color: var(--text-color); font-size: 1.1rem; line-height: 1.6;'>{result.get('feedback', '')}</div>", unsafe_allow_html=True)
        
    else:
        st.markdown(f"<div style='padding: 1.5rem; background-color: rgba(34, 197, 94, 0.1); border-left: 5px solid #22c55e; border-radius: 4px; color: var(--text-color); font-size: 1.1rem; line-height: 1.6;'>{result.get('feedback', result.get('overall_summary', ''))}</div>", unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # STRENGTHS & IMPROVEMENTS
    col_str, col_imp = st.columns(2)
    with col_str:
        st.markdown("<h4 style='color: #15803d;'>✅ Strengths</h4>", unsafe_allow_html=True)
        for s in result.get('strengths', []):
            st.markdown(f"<div style='margin-bottom: 0.5rem;'>• {s}</div>", unsafe_allow_html=True)
            
    with col_imp:
        st.markdown("<h4 style='color: #b91c1c;'>📉 Areas for Improvement</h4>", unsafe_allow_html=True)
        for i in result.get('improvements', result.get('areas_to_improve', [])):
            st.markdown(f"<div style='margin-bottom: 0.5rem;'>• {i}</div>", unsafe_allow_html=True)
            
        if mode == "Multi-Agent LLM" and result.get('misconceptions'):
            st.markdown("<h4 style='color: #b91c1c; margin-top: 1rem;'>⚠️ Misconceptions Detected</h4>", unsafe_allow_html=True)
            for m in result.get('misconceptions', []):
                st.markdown(f"<div style='margin-bottom: 0.5rem;'>• {m}</div>", unsafe_allow_html=True)
                
    st.divider()
    
    # RECOMMENDED ACTION
    st.markdown("### Recommended Learning Action", unsafe_allow_html=True)
    st.info(result.get('recommendation', ''))
    
    if mode == "Multi-Agent LLM" and result.get('practice_suggestions'):
        st.markdown("**Practice Suggestions:**")
        for p in result.get('practice_suggestions', []): 
            st.markdown(f"- {p}")
            
    if mode == "Multi-Agent LLM" and result.get('next_step'):
        st.markdown(f"**Next Step:** {result['next_step']}")
        
    st.divider()
    
    # TECHNICAL DETAILS (Less prominent)
    with st.expander("Technical & Research Details"):
        st.caption("This section contains internal metrics and backend configuration details for research validation.")
        
        # Engine details
        if mode == "Multi-Agent LLM":
            model_str = result.get('working_model', 'gemini-3.5-flash-lite')
            st.write(f"**Feedback Engine:** Gemini Multi-Agent ({model_str})")
        elif mode == "demo":
            st.write("**Feedback Engine:** Demonstration Mock")
        else:
            reason = result.get('fallback_reason', 'LLM service unavailable')
            st.write(f"**Feedback Engine:** Prototype Fallback (Reason: {reason})")
            
        st.markdown("---")
        
        # ML Details
        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("Predicted ML Score", f"{result.get('experimental_ml_score', 0):.2f} / 10")
        mc2.metric("Predicted ML Level", result.get("experimental_ml_learner_level", "Unknown"))
        mc3.metric("Models", "Sentence-BERT + XGBoost")
        
    st.markdown("<br>", unsafe_allow_html=True)
    
    # NAVIGATION
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Continue to Feedback Survey", type="primary", use_container_width=True):
            st.session_state.survey_completed = False
            navigate_to("evaluation")
            st.rerun()
    with col2:
        if st.button("View Full Dashboard", use_container_width=True):
            navigate_to("dashboard")
            st.rerun()

def page_feedback():
    # Deprecated page - route to results
    navigate_to("results")
    st.rerun()


# ============================================================
# PAGE: EVALUATION (STUDENT SURVEY)
# ============================================================

def page_evaluation():
    st.subheader("Student Feedback After Assessment")
    st.write("Please rate the feedback you received from 1 (Strongly Disagree) to 5 (Strongly Agree).")
    
    with st.form("evaluation_form"):
        r1 = st.slider("1. The feedback was relevant to my answer.", 1, 5, 3)
        r2 = st.slider("2. The feedback helped me understand my weakness.", 1, 5, 3)
        r3 = st.slider("3. The feedback was specific to my response.", 1, 5, 3)
        r4 = st.slider("4. The recommendation was useful.", 1, 5, 3)
        r5 = st.slider("5. The assessment was easy to use.", 1, 5, 3)
        
        useful_text = st.text_area("What was most useful about the feedback?")
        improve_text = st.text_area("What should be improved?")
        
        submitted = st.form_submit_button("Submit Survey & Finish", type="primary")
        
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
    st.subheader("Student Dashboard & Report")
    history = st.session_state.assessment_history
    
    if not history:
        st.info("No assessments completed yet. Start an assessment to see your dashboard.")
        return
        
    df = pd.DataFrame(history)
    
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Assessments", len(df))
    c2.metric("Average Score", f"{df['score'].mean():.2f}")
    c3.metric("Current Level", df.iloc[-1]['learner_level'])
    c4.metric("Last Topic", df.iloc[-1]['topic'])
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Performance by Topic")
        st.bar_chart(df.set_index("topic")["score"])
        
    with col2:
        st.markdown("### Recent Assessments")
        st.dataframe(df[["timestamp", "topic", "score", "learner_level"]], use_container_width=True)

    st.divider()
    
    st.markdown("### Detailed Session Review")
    for i, row in df.iterrows():
        with st.expander(f"{row['timestamp']} - {row['topic']} (Score: {row['score']:.2f})"):
            st.markdown(f"**Q1:** {row['question']}")
            st.markdown(f"> *{row['initial_response']}*")
            st.markdown(f"**Q2:** {row['followup_question']}")
            st.markdown(f"> *{row['followup_response']}*")
            st.markdown("---")
            st.markdown(f"**Feedback:** {row.get('feedback', row.get('overall_summary', ''))}")
            st.markdown(f"**Recommendation:** {row.get('recommendation', '')}")
            
    csv_data = df.to_csv(index=False).encode("utf-8")
    safe_name = st.session_state.current_student.strip().replace(" ", "_") or "student"
    
    st.download_button(
        label="📥 Download Your Personal Report",
        data=csv_data,
        file_name=f"Assessment_Report_{safe_name}.csv",
        mime="text/csv",
        type="primary"
    )

    st.divider()
    st.markdown("### Researcher Data Export")
    col1, col2 = st.columns(2)
    
    val_raw = Path("research/validation/student_validation_raw.csv")
    if val_raw.exists():
        with open(val_raw, "rb") as f:
            col1.download_button("📥 Download Validation Data (Raw CSV)", data=f, file_name="student_validation_raw.csv", mime="text/csv")
            
    val_survey = Path("research/validation/student_feedback_ratings.csv")
    if val_survey.exists():
        with open(val_survey, "rb") as f:
            col2.download_button("📥 Download Feedback Ratings (CSV)", data=f, file_name="student_feedback_ratings.csv", mime="text/csv")

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