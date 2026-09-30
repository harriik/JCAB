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
import speech_recognition as sr
import soundfile as sf
import io
from pathlib import Path

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
    page_title="JCAB | AI Assessment",
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
        padding: 1.5rem;
        background-color: #f8fafc;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        margin-bottom: 2rem;
    }
    .header-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0f172a;
        margin: 0;
    }
    .header-subtitle {
        font-size: 1.2rem;
        color: #334155;
        margin-top: 0.5rem;
    }
    .tech-badge {
        display: inline-block;
        background: #e2e8f0;
        color: #475569;
        padding: 4px 12px;
        border-radius: 16px;
        font-size: 0.85rem;
        margin-right: 8px;
        margin-top: 12px;
        font-weight: 600;
    }
    .chat-msg {
        padding: 1.2rem;
        border-radius: 8px;
        margin-bottom: 1.2rem;
        font-size: 1.1rem;
        line-height: 1.5;
    }
    .chat-ai {
        background-color: #f0fdf4;
        border-left: 5px solid #22c55e;
        color: #166534;
    }
    .chat-student {
        background-color: #eff6ff;
        border-left: 5px solid #3b82f6;
        color: #1e3a8a;
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
    st.title("🎓 JCAB Framework")
    st.caption("AI-Based Educational Assessment")
    
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

def transcribe_audio(audio_file) -> str:
    """Attempts to transcribe uploaded audio via SpeechRecognition."""
    try:
        audio_bytes = audio_file.read()
        audio_size = len(audio_bytes)
        logger.info(f"Audio size in bytes: {audio_size}")
        
        if audio_size == 0:
            return "Could not understand the audio. Please try again."
            
        audio_file.seek(0)
        
        # Parse and convert to 16-bit PCM WAV using soundfile
        data, samplerate = sf.read(io.BytesIO(audio_bytes))
        
        # If stereo, convert to mono for better STT
        if len(data.shape) > 1:
            data = data.mean(axis=1)
            
        duration = len(data) / samplerate if samplerate > 0 else 0
        logger.info(f"Audio parsed: sample_rate={samplerate}, length={len(data)} frames, duration={duration:.2f}s")
        
        if duration < 0.5:
            # Audio is too short
            return "Could not understand the audio. Please try again."
            
        # Write clean PCM WAV
        clean_wav_io = io.BytesIO()
        sf.write(clean_wav_io, data, samplerate, subtype='PCM_16', format='WAV')
        clean_wav_io.seek(0)
        
        r = sr.Recognizer()
        with sr.AudioFile(clean_wav_io) as source:
            audio_data = r.record(source)
            logger.info("Transcribing speech...")
            # Use 'en' as language configuration
            text = r.recognize_google(audio_data, language="en")
            logger.info("Transcription successful.")
            return text
            
    except sr.UnknownValueError:
        logger.warning("Transcription returned empty/unknown.")
        return "Could not understand the audio. Please try again."
    except sr.RequestError as e:
        logger.error(f"STT API Request Error: {e}")
        return f"[STT Service Error: {e}]"
    except Exception as e:
        logger.error(f"Transcription Failed: {e}", exc_info=True)
        return f"[Transcription Failed: {e}]"


# ============================================================
# PAGE: HOME
# ============================================================

def page_home():
    st.markdown(
        f"""
        <div class="header-box">
            <h1 class="header-title">Welcome to {APP_NAME}</h1>
            <div class="header-subtitle">
                An AI-powered conversational assessment framework that analyzes student responses and generates personalized learning feedback.
            </div>
            <div>
                <span class="tech-badge">Sentence-BERT</span>
                <span class="tech-badge">Machine Learning</span>
                <span class="tech-badge">XGBoost</span>
                <span class="tech-badge">Multi-Agent LLM</span>
                <span class="tech-badge">Conversational Assessment</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.info("### Evaluate\nEngage in a dynamic conversational assessment that adapts to your understanding.")
    with c2:
        st.success("### Analyze\nLeverage Sentence-BERT and XGBoost to perform deep semantic analysis of responses.")
    with c3:
        st.warning("### Feedback\nReceive highly personalized feedback to target learning gaps and concepts.")
        
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("Start New Assessment", type="primary", use_container_width=True):
            reset_assessment()
    with col2:
        if st.button("View Student Dashboard", use_container_width=True):
            navigate_to("dashboard")


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
                
                navigate_to("chat")
                st.rerun()


# ============================================================
# PAGE: CONVERSATIONAL CHAT
# ============================================================

def page_chat():
    st.subheader(f"Conversational Assessment ({st.session_state.selected_domain})")
    
    q_idx = st.session_state.current_q_index
    total_q = len(st.session_state.assessment_questions)
    
    if q_idx >= total_q or total_q == 0:
        # Failsafe if state is corrupted
        st.warning("No questions found. Returning to setup.")
        reset_assessment()
        return
        
    current_q_data = st.session_state.assessment_questions[q_idx]
    
    # Render Progress
    st.progress(max(0.01, q_idx / total_q), text=f"Question {q_idx + 1} / {total_q} | Topic: {current_q_data['topic']}")
    
    # Render AI Initial Question
    st.markdown(f"<div class='chat-msg chat-ai'>🤖 <b>AI Assessor:</b><br/>{current_q_data['question']}</div>", unsafe_allow_html=True)
    
    if st.session_state.current_stage == "initial":
        st.caption("🎙️ If microphone permission was denied, please allow access in browser settings. You can always type your response below.")
        
        # Audio Input
        audio_val = st.audio_input("Record Voice Response")
        if audio_val is not None:
            if st.session_state.last_audio != audio_val.file_id:
                with st.spinner("Transcribing audio..."):
                    transcript = transcribe_audio(audio_val)
                    st.session_state.initial_resp_text = transcript
                    st.session_state.last_audio = audio_val.file_id
                    
        # Text Input (bound to session state so STT updates it automatically)
        st.text_area("Your Response:", key="initial_resp_text", height=150, placeholder="Type your response here...")
        
        if st.button("Submit Response", type="primary"):
            if len(st.session_state.initial_resp_text.strip()) < 5:
                st.warning("Please provide a valid response.")
            else:
                st.session_state.initial_response = st.session_state.initial_resp_text
                st.session_state.current_stage = "followup"
                st.rerun()
                
    elif st.session_state.current_stage == "followup":
        # Show previous interaction
        st.markdown(f"<div class='chat-msg chat-student'>👤 <b>You:</b><br/>{st.session_state.initial_response}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-msg chat-ai'>🤖 <b>AI Follow-up:</b><br/>{current_q_data['followup']}</div>", unsafe_allow_html=True)
        
        st.caption("🎙️ If microphone permission was denied, please allow access in browser settings. You can always type your response below.")
        
        audio_val_2 = st.audio_input("Record Voice Response", key="audio_followup")
        if audio_val_2 is not None:
            if st.session_state.last_audio_2 != audio_val_2.file_id:
                with st.spinner("Transcribing audio..."):
                    transcript_2 = transcribe_audio(audio_val_2)
                    st.session_state.followup_resp_text = transcript_2
                    st.session_state.last_audio_2 = audio_val_2.file_id
                    
        st.text_area("Your Follow-up Response:", key="followup_resp_text", height=150, placeholder="Type your response here...")
        
        if st.button("Complete Concept", type="primary"):
            if len(st.session_state.followup_resp_text.strip()) < 5:
                st.warning("Please provide a valid response.")
            else:
                st.session_state.followup_response = st.session_state.followup_resp_text
                navigate_to("processing")
                st.rerun()

    st.divider()
    if st.button("End Assessment Early"):
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
                predicted_level_encoded,
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
            "predicted_score": predicted_score,
            "predicted_level": predicted_level,
        }
        
        # Merge all feedback keys safely
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
            "predicted_score": result["predicted_score"],
            "predicted_learner_level": result["predicted_level"],
            "feedback_mode": result.get("mode", "UNKNOWN"),
            "generated_feedback": result.get("feedback", ""),
            "recommended_learning_action": result.get("recommendation", "")
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
# PAGE: RESULTS
# ============================================================

def page_results():
    result = st.session_state.assessment_result
    
    st.subheader("Prototype ML Analysis")
    if st.session_state.demo_mode or not models_ready:
        st.warning("⚠️ Demonstration Output — not a measured research result.")
    else:
        st.caption("Preliminary Prototype Result — Generated using experimental ML models (XGBoost/Sentence-BERT).")
    
    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Score", f"{result['predicted_score']:.2f} / 10")
    c2.metric("Learner Level", result["predicted_level"])
    c3.metric("Concept Evaluated", result["topic"])
    
    st.divider()
    
    st.markdown("### Conceptual Analysis")
    col_str, col_imp = st.columns(2)
    with col_str:
        st.success("✅ **Strengths**")
        for s in result['strengths']:
            st.write(f"- {s}")
            
    with col_imp:
        st.error("📉 **Areas to Improve**")
        for i in result['improvements']:
            st.write(f"- {i}")
            
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("View Prototype Personalized Feedback", type="primary", use_container_width=True):
            navigate_to("feedback")
            st.rerun()
    with col2:
        if st.button("View Full Report / Dashboard", use_container_width=True):
            navigate_to("dashboard")
            st.rerun()


# ============================================================
# PAGE: FEEDBACK
# ============================================================

def page_feedback():
    result = st.session_state.assessment_result
    
    st.subheader("Prototype Personalized Feedback")
    
    mode = result.get('mode', 'Development Fallback (Rule-Based)')
    
    st.write(f"**Feedback Runtime Mode:** `{mode}`")
    
    if mode == "Multi-Agent LLM":
        model_str = result.get('working_model', 'gemini-3.5-flash')
        st.success(f"🟢 **Feedback Mode:** Multi-Agent LLM\n\n**Agents:** Response Analyzer • Personalized Feedback • Learning Recommendation • Coordinator\n\n**LLM Model:** `{model_str}`")
        engine_str = "Gemini Multi-Agent"
    elif mode == "demo":
        st.info("🔵 **Feedback Mode:** Demonstration Output")
        engine_str = "Demonstration Mock"
        model_str = "N/A"
    else:
        reason = result.get('fallback_reason', 'LLM service unavailable')
        st.warning(f"🟡 **Feedback Mode:** Prototype Fallback\n\n**Reason:** {reason}")
        engine_str = "Prototype Fallback"
        model_str = "N/A"
        
    with st.expander("Pipeline Details"):
        st.markdown(f"""
        - **Embedding:** Sentence-BERT (all-MiniLM-L6-v2)
        - **ML Analysis:** XGBoost
        - **Feedback Engine:** {engine_str}
        - **Agent Count:** 4
        - **LLM Model:** {model_str}
        """)
        
    st.markdown("---")
    st.markdown("### ML ANALYSIS")
    st.markdown(f"**Score:** {result['predicted_score']:.2f} / 10")
    st.markdown(f"**Learner Level:** {result['predicted_level']}")
    st.markdown("---")
    
    st.markdown("### PERSONALIZED FEEDBACK")
    
    if mode == "Multi-Agent LLM":
        st.markdown(f"**Overall Summary:** {result.get('overall_summary', '')}")
        st.markdown("**Strengths:**")
        for s in result.get('strengths', []): st.markdown(f"- {s}")
        st.markdown("**Concepts to Improve:**")
        for i in result.get('improvements', []): st.markdown(f"- {i}")
        st.markdown("**Misconceptions:**")
        for m in result.get('misconceptions', []): st.markdown(f"- {m}")
        st.markdown("**Personalized Explanation:**")
        st.write(result.get('feedback', ''))
        st.markdown("---")
        st.markdown("### LEARNING RECOMMENDATION")
        st.write(result.get('recommendation', ''))
        st.markdown("**Practice Suggestions:**")
        for p in result.get('practice_suggestions', []): st.markdown(f"- {p}")
        st.markdown("---")
        st.markdown("### NEXT STEP")
        st.info(result.get('next_step', ''))
    else:
        st.write(result.get('feedback', ''))
        st.markdown("**Strengths:**")
        for s in result.get('strengths', []): st.markdown(f"- {s}")
        st.markdown("**Areas to Improve:**")
        for i in result.get('improvements', []): st.markdown(f"- {i}")
        st.markdown("---")
        st.markdown("### LEARNING RECOMMENDATION")
        st.info(result.get('recommendation', ''))
    
    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Back to Analysis"):
            navigate_to("results")
            st.rerun()
    with c2:
        if st.button("Continue to Feedback Survey", type="primary", use_container_width=True):
            st.session_state.survey_completed = False
            navigate_to("evaluation")
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
    c2.metric("Average Score", f"{df['predicted_score'].mean():.2f}")
    c3.metric("Current Level", df.iloc[-1]['predicted_level'])
    c4.metric("Last Topic", df.iloc[-1]['topic'])
    
    st.divider()
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Performance by Topic")
        st.bar_chart(df.set_index("topic")["predicted_score"])
        
    with col2:
        st.markdown("### Recent Assessments")
        st.dataframe(df[["timestamp", "topic", "predicted_score", "predicted_level"]], use_container_width=True)

    st.divider()
    
    st.markdown("### Detailed Session Review")
    for i, row in df.iterrows():
        with st.expander(f"{row['timestamp']} - {row['topic']} (Score: {row['predicted_score']:.2f})"):
            st.markdown(f"**Q1:** {row['question']}")
            st.markdown(f"> *{row['initial_response']}*")
            st.markdown(f"**Q2:** {row['followup_question']}")
            st.markdown(f"> *{row['followup_response']}*")
            st.markdown("---")
            st.markdown(f"**Feedback:** {row['feedback']}")
            st.markdown(f"**Recommendation:** {row['recommendation']}")
            
    csv_data = df.to_csv(index=False).encode("utf-8")
    safe_name = st.session_state.current_student.strip().replace(" ", "_") or "student"
    
    st.download_button(
        label="📥 Download Your Personal Report",
        data=csv_data,
        file_name=f"JCAB_Report_{safe_name}.csv",
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