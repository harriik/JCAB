# JCAB - Java Conversational Assessment Bot
**Status: READY FOR GUIDE DEMO**

## 1. Project Title
Java Conversational Assessment Bot (JCAB) - A Multi-Agent LLM Approach to Semantic Concept Evaluation.

## 2. Project Objective
To evaluate students' conceptual understanding of Java programming through a conversational interface, using Sentence-BERT for semantic embeddings, XGBoost for predictive scoring, and a Multi-Agent LLM pipeline to generate personalized, targeted feedback and learning recommendations.

## 3. Current Implemented Architecture
Student 
↓ 
Conversational Java Assessment (Streamlit UI) 
↓ 
Sentence-BERT (Semantic Embeddings) 
↓ 
XGBoost (Teacher Score & Learner Level Prediction) 
↓ 
4 Specialized Agents using one Gemini LLM backend 
↓ 
Personalized Feedback + Recommendation 
↓ 
Dashboard/Report

**Important Note:**
*One Gemini LLM backend currently powers four specialized agents. Therefore the system is multi-agent, not multi-model.*

## 4. Technologies Used
- **Frontend/UI:** Streamlit
- **ML Pipeline:** SentenceTransformers (all-MiniLM-L6-v2), XGBoost, scikit-learn
- **LLM Integration:** Google GenAI SDK (gemini-3.5-flash / gemini-flash-latest / gemini-2.0-flash / gemini-1.5-flash)
- **Data Management:** Pandas, JSONL, CSV

## 5. Implemented Modules
- `app.py`: Streamlit User Interface and state management.
- `modules/assessment.py`: Defines the 50-question Java concept bank and domain logic.
- `modules/embeddings.py`: Generates dense vector representations of student conversations.
- `modules/prediction.py`: Applies pre-trained XGBoost models for rapid heuristic classification.
- `modules/llm_agents.py`: Orchestrates the 4-Agent LLM pipeline with adaptive rate limiting and failovers.
- `modules/feedback.py`: Wraps the pipeline and manages the robust rule-based Development Fallback.

## 6. Dataset Status
- **Real Research Data:** Separate and cleanly structured in `research/validation/student_validation_raw.csv`.
- **Synthetic Data:** Simulated pilot data (20 participants) successfully generated and maintained in `research/validation/synthetic/`.

## 7. ML Pipeline
- Embeddings and prediction inference take less than a second combined.
- Accurate scaling across 5 classification levels (`Excellent`, `Good`, `Partial`, `Weak`, `Incorrect`).

## 8. Multi-Agent LLM Pipeline
- **Agent 1 (Response Analyzer):** Determines structural weaknesses, specific strengths, and conceptual misunderstandings.
- **Agent 2 (Personalized Feedback):** Takes Agent 1's context to draft targeted, conversational feedback.
- **Agent 3 (Learning Recommendation):** Recommends actionable next steps based on Agent 1 and 2 output.
- **Agent 4 (Coordinator):** Synthesizes outputs into the final strictly formatted response.
- Includes dynamic model caching, exponential backoff (for 503/429 limits), and a 45-second network timeout tolerance to maximize uptime on free-tier APIs.

## 9. Validation Framework
- Configured to log interactions cleanly. 
- Discards test runs (e.g. `TEST_01`) dynamically to preserve research validity.

## 10. Synthetic Pilot Simulation
- Completed. Successfully demonstrates the pipeline capabilities on 64 simulated interactions without polluting the real validation datasets.

## 11. Current Limitations
- **API Quota Restrictions:** Free-tier Gemini is highly constrained (15 RPM). The pipeline is stabilized around this using a resilient multi-tier fallback architecture, but may still revert to rule-based fallback if multiple models exhaust their quota during heavy use.
- **Human participant validation has not yet been conducted.**

## 12. Pending Work
- Formal launch of the MCA pilot with actual students.

## 13. Exact steps for demonstrating the application
1. Launch the application: `.\.venv\Scripts\python.exe -m streamlit run app.py`
2. Enter the **Participant ID** as `TEST_01` (to exclude your test run from the final research dataset).
3. Check the Research Consent box.
4. Go to **Domain Selection** and choose **Java**.
5. Respond to the **Assessment Question** using text.
6. Submit and respond to the **Follow-up Question**.
7. Wait during the **Processing** phase. The UI will explicitly display the Multi-Agent tracking:
   - `Generating personalized feedback...`
   - `✓ Response Analyzer`
   - `✓ Personalized Feedback`
   - `✓ Learning Recommendation`
   - `⏳ Coordinator`
8. Review the final Dashboard, which confirms:
   - The ML Score and Learner Level.
   - The **Feedback Mode**: "Multi-Agent LLM" (or "Development Fallback (Rule-Based)" if network constraints blocked the LLM).
   - The agents' generated strengths, improvements, and recommendations.
