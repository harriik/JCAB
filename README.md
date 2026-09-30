# JCAB: Java Conversational Assessment and Personalized Feedback System

## Project Overview
This project implements an Adaptive Personalized Feedback Framework for Enhancing Multi-Agent Large Language Model-Based Conversational Assessment in Education. It is a preliminary academic research prototype that uses semantic embeddings (Sentence-BERT) and a machine learning pipeline (XGBoost) to predict learner performance on conceptual programming topics.

The application allows students to engage in a dynamic conversation with an AI assessor, providing both text and voice inputs.

## Research Architecture
- **Language/Domain:** Java Programming (Python, C++, DBMS, OS, Networks are coming soon).
- **Core Pipeline:** Text/Audio -> Semantic Embedding -> Score/Level Prediction -> Multi-factor Feedback Generation.
- **ML Models:** XGBoost Classifier (Learner Level), XGBoost Regressor (Score).
- **Embeddings:** `all-MiniLM-L6-v2` (Sentence-BERT, 384 dimensions).
- **Dataset:** 50-topic Java conversational dataset (Synthetic Development).

## UI Modules
The system is built as a state-driven Streamlit application featuring:
1. **Home:** Introduction and overview of the platform.
2. **Setup:** Student profile registration and domain selection.
3. **Configuration:** Dynamic generation of assessment questions from the question bank.
4. **Chat Interface:** Conversational UI capturing initial responses and follow-up interactions (supports both text and native browser microphone).
5. **Processing:** Pipeline execution.
6. **Results & Analysis:** Detailed breakdown of score, learner level, and conceptual strengths/weaknesses.
7. **Personalized Feedback:** Structured feedback indicating the next best learning step.
8. **Dashboard:** Progress tracking and historical visualization.
9. **Report:** CSV export functionality.

## Demo Mode
In the sidebar, a **Demo Mode** toggle is available. Activating it safely marks generated results as "Demonstration Output" to clarify when presentations are being made without live backends.

## How to Run
1. Ensure you have Python 3.9+ (using the local `.venv`).
2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```

## Voice Input Setup
Voice input uses Streamlit's native `st.audio_input` component. The transcription is processed locally via the `SpeechRecognition` library (which uses the Google Web Speech API). If you lack microphone permissions or if transcription fails, the system automatically degrades gracefully to the manual text-area input fallback. No extra keys are required.

## Known Limitations
- The underlying ML models were trained on synthetic development data; predictions should not be treated as final research validation.
- Non-Java domains are visible in the UI but structurally restricted from starting.
- Multi-Agent LLM generation for dynamic follow-ups is currently simulated by the expanded question bank while the real LLM endpoint is under development.
