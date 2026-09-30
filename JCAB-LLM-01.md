# JCAB-LLM-01: Multi-Agent LLM Feedback Layer Implementation

## Overview
This document outlines the implementation of the Multi-Agent LLM Feedback Layer for the JCAB (Java Conversational Assessment Bot) research project.

## Multi-Agent Architecture
The system employs a 4-agent pipeline to generate robust, adaptive, and highly personalized feedback for students, utilizing the analytical grounding provided by the upstream XGBoost ML models.

### Agent Responsibilities & Inputs/Outputs

1. **Agent 1: Response Analyzer**
   - **Responsibility:** Diagnose conceptual understanding.
   - **Input:** Question, responses, XGBoost score, Learner Level, Topic.
   - **Output:** Strengths, weaknesses, misconceptions, confidence score.
   
2. **Agent 2: Personalized Feedback Agent**
   - **Responsibility:** Generate the personalized feedback prose tailored to the specific student's ML score and Agent 1's conceptual diagnosis.
   - **Input:** Agent 1 Output, Original Q&A, XGBoost score, Learner Level.
   - **Output:** Summary, strengths to highlight, areas to improve, explanatory prose.
   
3. **Agent 3: Learning Recommendation Agent**
   - **Responsibility:** Generate an actionable learning plan based specifically on identified weaknesses.
   - **Input:** Agent 1 and Agent 2 Outputs, Topic, XGBoost score, Learner Level.
   - **Output:** Recommended topic, specific action, practice suggestions, next step.

4. **Agent 4: Feedback Coordinator**
   - **Responsibility:** Synthesize the preceding outputs into a cohesive final JSON payload matching the UI schema, preventing contradictions or hallucinations.
   - **Input:** Outputs from Agents 1, 2, and 3, plus ML context.
   - **Output:** Final structured JSON strictly mapping to the UI layout.

## LLM Provider Configuration
The framework utilizes the `google-genai` SDK with the model `gemini-2.5-flash` to execute the pipeline.
- It abstracts the initialization behind a robust `LLMProvider` class.
- The provider is driven securely via environment variables: `LLM_API_KEY`, `LLM_PROVIDER`, and `LLM_MODEL`.
- No API keys are hardcoded in the source code.

## Fallback Mechanism
If the `LLM_API_KEY` is not present, or if the API suffers a timeout, rate limit, or failure to parse the JSON schema:
1. The pipeline catches the exception.
2. The UI gracefully fails over to the `Development Fallback (Rule-Based)` feedback.
3. The UI explicitly labels this as "Prototype Feedback — LLM service unavailable." to prevent deceiving the user into believing the rule-based output was LLM-generated.

## Personalization Mechanism
Adaptivity is heavily driven by the upstream **XGBoost Predicted Score** (0-10 scale). 
- The LLM agents are explicitly instructed in their prompts to branch their tone and depth based on the score. 
- High scores trigger advanced conceptual pushes; low scores trigger fundamental simplified explanations and misconception correction.

## Current Limitations
- Without a live API key in the deployment environment, the LLM pipeline defaults to the rule-based fallback.
- Context window size is currently sufficient for single-topic assessments but may need chunking if full-course conversations are implemented.
