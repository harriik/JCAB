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
import streamlit as st

# Local modules
from config.settings import (
    APP_NAME, APP_VERSION, EMBEDDING_MODEL_NAME, EXPECTED_EMBEDDING_DIMENSION,
    TOPICS, DATASET_STATUS, RESEARCH_STATUS, TARGET_POPULATION, USE_LLM_FEEDBACK
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
from modules.assessment import get_assessment_questions

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
    page_title="JCAB | Academic Prototype",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Clean, professional styling suitable for an academic prototype
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    .header-box {
        padding: 1.5rem;
        border-bottom: 2px solid #e2e8f0;
        margin-bottom: 2rem;
    }
    .header-title {
        font-size: 2rem;
        font-weight: 600;
        color: #1e293b;
        margin: 0;
    }
    .header-subtitle {
        font-size: 1.1rem;
        color: #475569;
        margin-top: 0.5rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. SESSION STATE
# ============================================================

DEFAULT_SESSION_STATE = {
    "assessment_result": None,
    "assessment_history": [],
    "current_stage": "setup", # setup, initial, followup, results
    "current_student": "",
    "current_id": "",
    "current_topic": "",
    "current_question": "",
    "current_followup_question": "",
    "initial_response": "",
    "followup_response": ""
}

for key, default_value in DEFAULT_SESSION_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


def reset_assessment():
    """Resets the assessment flow to the beginning."""
    st.session_state.current_stage = "setup"
    st.session_state.initial_response = ""
    st.session_state.followup_response = ""
    st.session_state.assessment_result = None

# ============================================================
# 4. INITIALIZE MODELS
# ============================================================

models_ready = False
model_error = None

try:
    with st.spinner("Loading ML models..."):
        embedding_model = load_embedding_model()
        score_model, level_model = load_ml_models()
    models_ready = True
except Exception as exc:
    model_error = str(exc)
    embedding_model = None
    score_model = None
    level_model = None
    logger.exception("Model initialization failed")


# ============================================================
# 5. HEADER & SIDEBAR
# ============================================================

st.markdown(
    """
    <div class="header-box">
        <h1 class="header-title">JCAB Academic Prototype</h1>
        <div class="header-subtitle">
            Java Conversational Assessment and Personalized Feedback System
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

with st.sidebar:
    st.markdown("### Research Metadata")
    st.write(f"**Domain:** Java Programming")
    st.write(f"**Target:** {TARGET_POPULATION}")
    st.write(f"**Status:** {RESEARCH_STATUS}")
    
    st.info(f"**Dataset:** {DATASET_STATUS}")
    st.caption("Note: Results reflect preliminary development and have not completed final validation.")
    
    st.divider()
    
    st.markdown("### System Status")
    if models_ready:
        st.success("Models Loaded")
    else:
        st.error("Models Offline")
    
    st.divider()
    st.caption(f"{APP_NAME} v{APP_VERSION}")


if not models_ready:
    st.error("### Application cannot start")
    st.write("One or more preliminary trained models could not be loaded.")
    if model_error:
        st.error(f"Model loading error:\n\n{model_error}")
    st.stop()


# ============================================================
# 6. STAGE 1: ASSESSMENT SETUP
# ============================================================

if st.session_state.current_stage == "setup":
    st.subheader("1. Assessment Setup")
    
    with st.form("setup_form"):
        col1, col2 = st.columns(2)
        with col1:
            student_name = st.text_input("Student Name", value=st.session_state.current_student)
            student_id = st.text_input("Student ID (Optional)", value=st.session_state.current_id)
        with col2:
            topic = st.selectbox("Select Topic", TOPICS)
            
        submitted = st.form_submit_button("Start Assessment")
        
        if submitted:
            if not student_name.strip():
                st.warning("Please enter a student name.")
            else:
                st.session_state.current_student = student_name
                st.session_state.current_id = student_id
                st.session_state.current_topic = topic
                
                # Fetch questions from controlled bank
                q, fq = get_assessment_questions(topic)
                st.session_state.current_question = q
                st.session_state.current_followup_question = fq
                
                st.session_state.current_stage = "initial"
                st.rerun()

# ============================================================
# 7. STAGE 2: INITIAL RESPONSE
# ============================================================

elif st.session_state.current_stage == "initial":
    st.subheader("2. Initial Assessment")
    
    st.info(f"**Topic:** {st.session_state.current_topic}")
    st.markdown(f"**Question:** {st.session_state.current_question}")
    
    initial_resp = st.text_area("Your Answer:", height=150)
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Cancel"):
            reset_assessment()
            st.rerun()
    with col2:
        if st.button("Submit Answer", type="primary"):
            if len(initial_resp.strip()) < 10:
                st.warning("Response is too short for meaningful analysis. Please elaborate.")
            else:
                st.session_state.initial_response = initial_resp
                st.session_state.current_stage = "followup"
                st.rerun()

# ============================================================
# 8. STAGE 3: FOLLOW-UP RESPONSE
# ============================================================

elif st.session_state.current_stage == "followup":
    st.subheader("3. Follow-up Interaction")
    
    st.info(f"**Initial Answer:** {st.session_state.initial_response}")
    st.markdown(f"**Follow-up Question:** {st.session_state.current_followup_question}")
    
    followup_resp = st.text_area("Your Response to Follow-up:", height=150)
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("Cancel"):
            reset_assessment()
            st.rerun()
    with col2:
        if st.button("Complete Assessment", type="primary"):
            if len(followup_resp.strip()) < 10:
                st.warning("Response is too short for meaningful analysis. Please elaborate.")
            else:
                st.session_state.followup_response = followup_resp
                st.session_state.current_stage = "processing"
                st.rerun()

# ============================================================
# 9. STAGE 4: PROCESSING & RESULTS
# ============================================================

elif st.session_state.current_stage == "processing":
    st.subheader("Processing Assessment...")
    
    try:
        # Build conversation
        conversation_text = build_conversation(
            st.session_state.initial_response,
            st.session_state.followup_response
        )
        
        # Embeddings
        embedding = generate_embedding(conversation_text, embedding_model)
        
        # Predictions
        predicted_score = predict_score(embedding, score_model)
        predicted_level_encoded = predict_level(embedding, level_model)
        predicted_level = decode_learner_level(predicted_level_encoded)
        
        # Feedback
        feedback_data = get_feedback(
            st.session_state.current_question,
            st.session_state.initial_response,
            st.session_state.current_followup_question,
            st.session_state.followup_response,
            predicted_score,
            predicted_level_encoded,
            st.session_state.current_topic,
            use_llm=USE_LLM_FEEDBACK
        )
        
        result = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "student_name": st.session_state.current_student,
            "student_id": st.session_state.current_id,
            "topic": st.session_state.current_topic,
            "question": st.session_state.current_question,
            "initial_response": st.session_state.initial_response,
            "followup_question": st.session_state.current_followup_question,
            "followup_response": st.session_state.followup_response,
            "predicted_score": predicted_score,
            "predicted_level_encoded": predicted_level_encoded,
            "predicted_level": predicted_level,
            "embedding_dimension": embedding.shape[1],
            "strengths": feedback_data["strengths"],
            "improvements": feedback_data["improvements"],
            "feedback": feedback_data["feedback"],
            "recommendation": feedback_data["recommendation"],
            "feedback_mode": feedback_data.get("mode", "Development")
        }
        
        st.session_state.assessment_result = result
        st.session_state.assessment_history.append(result)
        st.session_state.current_stage = "results"
        st.rerun()
        
    except Exception as exc:
        logger.exception("Assessment execution failed")
        st.error("An error occurred during processing.")
        st.exception(exc)
        if st.button("Return to Setup"):
            reset_assessment()
            st.rerun()

elif st.session_state.current_stage == "results":
    
    if st.button("Start New Assessment", type="primary"):
        reset_assessment()
        st.rerun()
        
    result = st.session_state.assessment_result
    
    st.subheader("4. Assessment Results")
    st.caption("Results are generated using preliminary models trained on synthetic data.")
    
    # Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Predicted Score", f"{result['predicted_score']:.2f} / 10")
    c2.metric("Predicted Learner Level", result["predicted_level"])
    c3.metric("Feedback Engine", result["feedback_mode"])
    
    st.divider()
    
    # Feedback
    col_str, col_imp = st.columns(2)
    with col_str:
        st.markdown("#### Strengths")
        for s in result['strengths']:
            st.write(f"- {s}")
            
    with col_imp:
        st.markdown("#### Areas for Improvement")
        for i in result['improvements']:
            st.write(f"- {i}")
            
    st.markdown("#### Personalized Feedback")
    st.info(result['feedback'])
    
    st.markdown("#### Recommended Practice")
    st.success(result['recommendation'])
    
    st.divider()
    
    # Expandable Details
    with st.expander("View Full Conversation"):
        st.markdown("**Initial Question:**")
        st.write(result['question'])
        st.markdown("**Initial Response:**")
        st.write(f"> {result['initial_response']}")
        st.markdown("**Follow-up Question:**")
        st.write(result['followup_question'])
        st.markdown("**Follow-up Response:**")
        st.write(f"> {result['followup_response']}")
        
    with st.expander("Technical Architecture Details"):
        st.write(f"**Embedding Model:** {EMBEDDING_MODEL_NAME}")
        st.write(f"**Embedding Dimension:** {result['embedding_dimension']}")
        st.write(f"**Score Model:** XGBoost Regressor")
        st.write(f"**Class Model:** XGBoost Classifier")
        st.write(f"**Decoded Learner Level:** {result['predicted_level']}")
        st.write("**Data Source:** Synthetic Development Dataset")
        st.write("**Note:** Score and learner level are independent model outputs.")
        
    # Download Report
    report_df = pd.DataFrame([result])
    csv_data = report_df.to_csv(index=False).encode("utf-8")
    safe_name = result["student_name"].strip().replace(" ", "_") or "student"
    
    st.download_button(
        label="Download Assessment Report (CSV)",
        data=csv_data,
        file_name=f"JCAB_Assessment_{safe_name}.csv",
        mime="text/csv"
    )