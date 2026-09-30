# JCAB Multi-Agent LLM Pipeline Architecture

## Data Flow
The pipeline follows a sequential enrichment approach, utilizing ML metrics to ground the LLM's generative feedback:

1. **Input Layer**
   - Student interacts with the UI, answering the initial question and follow-up.
   
2. **Embedding & ML Layer**
   - `Sentence-BERT (all-MiniLM-L6-v2)` constructs a 384-dimensional semantic embedding from: `Initial_Response + " [SEP] " + Followup_Response`.
   - `XGBoost Regressor` outputs the predicted conceptual score (0-10).
   - `XGBoost Classifier` outputs the encoded learner level (e.g. Advanced, Beginner).

3. **Multi-Agent LLM Layer**
   - **Agent 1 (Response Analyzer):** Ingests the raw Q&A, topic, and ML metrics. Outputs a structured conceptual diagnosis identifying strengths, weaknesses, and misconceptions.
   - **Agent 2 (Personalized Feedback):** Ingests the output of Agent 1 and the ML metrics. Generates personalized prose explaining exactly what the student got right and wrong.
   - **Agent 3 (Learning Recommendation):** Ingests the analysis and feedback from Agents 1 and 2. Outputs a highly targeted recommendation and practice steps based specifically on the identified weaknesses.
   - **Agent 4 (Feedback Coordinator):** Ingests all previous outputs and synthesizes them into a final structured JSON matching the UI's display requirements without hallucinating new claims.

4. **Output & Fallback Layer**
   - If the LLM pipeline succeeds, the Coordinator's JSON populates the UI's "Prototype Personalized Feedback" sections.
   - If the LLM provider fails (e.g., API key missing, rate limits), the pipeline is entirely bypassed, and a robust `Development Fallback (Rule-Based)` mechanism provides fallback feedback ensuring the assessment flow is never broken.

## LLM Provider Abstraction
The system utilizes `LLMProvider` which relies on environmental variables (`LLM_PROVIDER`, `LLM_API_KEY`, `LLM_MODEL`) to allow dynamic swapping of backend models (e.g., Gemini). Outputs are strongly typed utilizing `pydantic` schemas.
