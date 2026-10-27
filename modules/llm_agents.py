"""
Multi-Agent LLM Feedback Layer for JCAB.
Implements the 4-agent architecture:
1. Response Analyzer
2. Personalized Feedback Agent
3. Learning Recommendation Agent
4. Feedback Coordinator
"""

import os
import time
import json
import logging
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger("JCAB_LLM")

# Define structured output schemas using Pydantic
class Agent1Analysis(BaseModel):
    understanding: str = Field(description="One concise sentence (maximum 15 words) summarizing actual conceptual understanding.")
    strengths: List[str] = Field(default_factory=list, description="Maximum 2 concise items (5-15 words each) identifying what the student understood correctly. Empty list if none.")
    knowledge_gaps: List[str] = Field(default_factory=list, description="Maximum 2 concise items (5-15 words each) identifying specific gaps. Empty list if none.")
    weaknesses: List[str] = Field(default_factory=list, description="Alias for knowledge_gaps (maximum 2 concise items, 5-15 words each).")
    misconceptions: List[str] = Field(default_factory=list, description="Maximum 2 items (5-15 words each), ONLY when the student explicitly stated a false concept. Empty list if none.")
    missing_concepts: List[str] = Field(default_factory=list, description="Maximum 2 items (5-15 words each) identifying key omitted concepts. Empty list if none.")
    confidence: float = Field(default=0.9, description="Confidence score between 0.0 and 1.0.")

class Agent2Feedback(BaseModel):
    feedback_summary: str = Field(description="One concise sentence (maximum 20 words) summarizing feedback.")
    strengths: List[str] = Field(default_factory=list, description="Maximum 2 concise items (5-15 words each) highlighting demonstrated strengths.")
    areas_to_improve: List[str] = Field(default_factory=list, description="Maximum 2 concise items (5-15 words each) highlighting priority areas for improvement.")
    explanation: str = Field(description="Personalized educational explanation: strictly 2-4 sentences, MAXIMUM 60 WORDS. Direct and constructive. No flattery. MCA student level.")
    encouragement: str = Field(default="", description="Optional brief closing (maximum 10 words), avoid flattering adjectives.")

class Agent3Recommendation(BaseModel):
    recommended_topic: str = Field(description="Specific Java topic for next study step.")
    reason: str = Field(description="One concise sentence (maximum 15 words) explaining why.")
    recommended_action: str = Field(description="One concrete next action sentence of MAXIMUM 20 WORDS.")
    practice_suggestions: List[str] = Field(default_factory=list, description="Maximum 2 concrete practice exercises. Each MUST be one sentence of MAXIMUM 20 WORDS.")
    next_learning_step: str = Field(default="", description="Immediate next action: exactly one sentence of MAXIMUM 20 WORDS.")

class Agent4Coordinator(BaseModel):
    overall_summary: str = Field(description="One concise sentence (strictly MAXIMUM 20 WORDS) stating the overall assessment result. Do not repeat verbatim in personalized_feedback.")
    strengths: List[str] = Field(default_factory=list, description="Maximum 2 concise items (5-15 words each) highlighting demonstrated strengths. Empty list if none.")
    areas_to_improve: List[str] = Field(default_factory=list, description="Maximum 2 concise items (5-15 words each) highlighting priority areas for improvement.")
    conceptual_issues: List[str] = Field(default_factory=list, description="0 to 2 items (5-15 words each), ONLY if explicit misconceptions were identified. Empty list otherwise.")
    personalized_feedback: str = Field(description="Personalized feedback paragraph: strictly 2-4 sentences, MAXIMUM 60 WORDS. Direct, constructive, educational. Must NOT repeat overall_summary.")
    learning_recommendation: str = Field(description="One concrete recommendation sentence of MAXIMUM 20 WORDS.")
    next_learning_step: str = Field(default="", description="One concrete next action sentence of MAXIMUM 20 WORDS.")
    practice_suggestions: List[str] = Field(default_factory=list, description="Maximum 2 actionable practice exercises (each one sentence of MAXIMUM 20 WORDS).")

class LLMProvider:
    """Clean abstraction for LLM API providers."""
    
    _last_request_time = 0.0
    _total_requests = 0
    _total_retries = 0
    _working_model = None

    def __init__(self):
        # Resolve API key: check environment variables first, then Streamlit secrets
        api_key = os.environ.get("LLM_API_KEY", "") or os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            try:
                import streamlit as st
                if hasattr(st, "secrets"):
                    if "LLM_API_KEY" in st.secrets:
                        api_key = str(st.secrets["LLM_API_KEY"]).strip()
                    elif "GEMINI_API_KEY" in st.secrets:
                        api_key = str(st.secrets["GEMINI_API_KEY"]).strip()
            except Exception:
                pass
        self.api_key = api_key
        if api_key and "LLM_API_KEY" not in os.environ:
            os.environ["LLM_API_KEY"] = api_key

        # Resolve model name: environment variable, then Streamlit secrets, then default
        model = os.environ.get("LLM_MODEL", "")
        if not model:
            try:
                import streamlit as st
                if hasattr(st, "secrets") and "LLM_MODEL" in st.secrets:
                    model = str(st.secrets["LLM_MODEL"]).strip()
            except Exception:
                pass
        self.model = model or "gemini-3.5-flash-lite"

        # Resolve provider
        provider = os.environ.get("LLM_PROVIDER", "")
        if not provider:
            try:
                import streamlit as st
                if hasattr(st, "secrets") and "LLM_PROVIDER" in st.secrets:
                    provider = str(st.secrets["LLM_PROVIDER"]).strip()
            except Exception:
                pass
        self.provider = (provider or "gemini").lower()
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY environment variable or Streamlit secret is not set.")
            
        if self.provider == "gemini":
            from google import genai
            from google.genai import types
            self.client = genai.Client(api_key=self.api_key, http_options={'timeout': 60000.0})
            self.types = types
            # Static fallback list to avoid unnecessary API calls during initialization
            fallback_models = ["gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]
            self.candidate_models = [self.model]
            for m in fallback_models:
                if m != self.model:
                    self.candidate_models.append(m)
                
            # Prioritize the currently known working model if one exists
            if LLMProvider._working_model:
                if LLMProvider._working_model in self.candidate_models:
                    self.candidate_models.remove(LLMProvider._working_model)
                self.candidate_models.insert(0, LLMProvider._working_model)
        else:
            raise NotImplementedError(f"Provider {self.provider} is not supported yet.")

    def generate_structured_content(self, prompt: str, schema_class: type[BaseModel]) -> BaseModel:
        import time
        if self.provider == "gemini":
            last_err = None
            
            # Dynamically sort models for this request to respect working model
            models_to_try = list(self.candidate_models)
            if LLMProvider._working_model and LLMProvider._working_model in models_to_try:
                models_to_try.remove(LLMProvider._working_model)
                models_to_try.insert(0, LLMProvider._working_model)
                
            for candidate_model in models_to_try:
                chat = self.client.chats.create(
                    model=candidate_model,
                    config=self.types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=schema_class,
                        temperature=0.2,
                        max_output_tokens=600,
                    )
                )
                
                for attempt in [1, 2, 3, 4]:
                    # Adaptive rate limit: 4 seconds between requests (15 RPM free tier limit)
                    now = time.time()
                    elapsed = now - LLMProvider._last_request_time
                    if elapsed < 4.0:
                        wait_time = 4.0 - elapsed
                        logger.info(f"Adaptive rate limit: Waiting {wait_time:.2f}s before next request.")
                        time.sleep(wait_time)
                    
                    LLMProvider._last_request_time = time.time()
                    LLMProvider._total_requests += 1
                    
                    try:
                        # google-genai does not accept timeout directly here in simple calls,
                        # but we can rely on the SDK's internal timeouts.
                        response = chat.send_message(prompt)
                        ans = schema_class.model_validate_json(response.text)
                        
                        if LLMProvider._working_model != candidate_model:
                            logger.info(f"Successfully switched working model to {candidate_model}")
                            LLMProvider._working_model = candidate_model
                            
                        return ans
                    except Exception as e:
                        err_str = str(e).lower()
                        last_err = e
                        
                        # Network Errors
                        if "timeout" in err_str or "timed out" in err_str or "connection" in err_str or "winerror" in err_str or "getaddrinfo" in err_str or "ssl" in err_str:
                            if attempt < 4:
                                logger.warning(f"Network error on {candidate_model}. Retry {attempt}/3 without switching model.")
                                LLMProvider._total_retries += 1
                                time.sleep(2.0)  # Short wait for network blip
                                continue
                            else:
                                logger.error(f"Network error on {candidate_model} persisted. Stopping failover chain.")
                                raise e # Do not failover for general network issues
                        
                        # Quota Errors (429)
                        elif "429" in err_str or "quota" in err_str or "resource_exhausted" in err_str:
                            if attempt < 4:
                                backoff = 2 ** attempt  # 2s, 4s, 8s
                                logger.warning(f"Quota exhausted (429) on {candidate_model}. Waiting {backoff}s...")
                                LLMProvider._total_retries += 1
                                time.sleep(backoff)
                                continue
                            else:
                                logger.error(f"Quota exhausted on {candidate_model} permanently. Stopping failover chain.")
                                raise e
                                
                        # Transient Server Errors (50x)
                        elif "503" in err_str or "502" in err_str or "504" in err_str or "500" in err_str or "unavailable" in err_str or "high demand" in err_str:
                            if attempt < 4:
                                backoff = 2 ** attempt
                                logger.warning(f"Transient server error (50x) on {candidate_model}. Waiting {backoff}s...")
                                LLMProvider._total_retries += 1
                                time.sleep(backoff)
                                continue
                            else:
                                logger.error(f"Model {candidate_model} is unavailable. Stopping failover chain.")
                                raise e
                                
                        # Model Errors (404/400)
                        elif "404" in err_str or "not_found" in err_str:
                            logger.warning(f"Model {candidate_model} not found (404). Failing over to next model.")
                            break # Move to next model immediately
                        elif "400" in err_str:
                            if "not supported" in err_str or "modalities" in err_str:
                                logger.warning(f"Model {candidate_model} is incompatible (400). Failing over.")
                                break # Move to next model immediately
                            else:
                                raise e # other 400 error
                                
                        # Auth Errors (401/403)
                        elif "401" in err_str or "403" in err_str:
                            logger.error(f"Authentication/Configuration error ({err_str}).")
                            raise e # fatal auth/config error
                            
                        # Unknown Errors
                        else:
                            raise e
            
            raise Exception(f"All candidate models failed or unavailable. Last error: {last_err}")


def sanitize_text(text: str) -> str:
    """Removes excessive flattery phrases and normalizes whitespace."""
    if not text:
        return ""
    import re
    flattery_phrases = [
        r"your response is exceptionally well-crafted",
        r"exceptionally well-crafted",
        r"you have a fantastic grasp of the fundamentals",
        r"fantastic grasp of the fundamentals",
        r"keep pushing your knowledge further",
        r"demonstrates a complete understanding of how the system operates under the hood",
        r"under the hood",
        r"outstanding response",
        r"stellar job",
        r"brilliant explanation",
        r"kudos on",
    ]
    cleaned = text.strip()
    for phrase in flattery_phrases:
        cleaned = re.sub(phrase, "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    cleaned = re.sub(r'^\s*[,;.-]\s*', '', cleaned)
    return cleaned

def clamp_words_by_sentence(text: str, max_words: int) -> str:
    """
    Limits text to max_words without breaking sentences mid-way.
    If multiple sentences exist, retains complete sentences that fit within max_words.
    If the first sentence itself exceeds max_words, cleanly cuts at word boundary with a period.
    """
    if not text:
        return ""
    text = sanitize_text(text)
    words = text.split()
    if len(words) <= max_words:
        return text

    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    valid_sentences = []
    current_count = 0
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        s_count = len(s_clean.split())
        if current_count + s_count <= max_words:
            valid_sentences.append(s_clean)
            current_count += s_count
        else:
            break

    if valid_sentences:
        return " ".join(valid_sentences).strip()

    truncated = " ".join(words[:max_words]).rstrip(",;: -")
    if not truncated.endswith((".", "!", "?")):
        truncated += "."
    return truncated

def clamp_list_items(items: List[str], max_items: int = 2, max_words_per_item: int = 20) -> List[str]:
    """Clamps list to max_items, removes duplicate entries, and caps words per item."""
    if not items:
        return []
    result = []
    seen = set()
    for item in items:
        if not item or not isinstance(item, str):
            continue
        cleaned = clamp_words_by_sentence(item.strip(), max_words=max_words_per_item)
        if not cleaned:
            continue
        norm_key = cleaned.lower()
        if norm_key not in seen:
            seen.add(norm_key)
            result.append(cleaned)
        if len(result) >= max_items:
            break
    return result

def deduplicate_summary_and_feedback(summary: str, feedback: str) -> str:
    """Ensures summary is not repeated verbatim at the start of or inside personalized feedback."""
    if not summary or not feedback:
        return feedback
    import re
    norm_summary = re.sub(r'[^a-zA-Z0-9\s]', '', summary.lower()).strip()
    sentences = re.split(r'(?<=[.!?])\s+', feedback.strip())
    filtered = []
    for s in sentences:
        norm_s = re.sub(r'[^a-zA-Z0-9\s]', '', s.lower()).strip()
        if norm_s and norm_s == norm_summary:
            continue
        filtered.append(s)
    if filtered:
        return " ".join(filtered).strip()
    return feedback

def normalize_and_validate_feedback(a4: Agent4Coordinator, context: Dict[str, Any]) -> Dict[str, Any]:
    """
    Applies field-specific validation and normalization to enforce strict limits:
    - Overall summary: 1 sentence, max 20 words
    - Strengths: max 2 items, max 15 words each
    - Improvements: max 2 items, max 15 words each
    - Misconceptions: max 2 items, max 15 words each (empty if none)
    - Personalized feedback: max 60 words (2-4 sentences, no verbatim duplication)
    - Recommended next steps: max 2 items, max 20 words each
    """
    topic = context.get("topic", "Java")
    score = context.get("score", 5.0)

    # 1. Overall Summary (max 20 words)
    summary = clamp_words_by_sentence(a4.overall_summary, max_words=20)
    if not summary:
        if score >= 8.5:
            summary = f"Demonstrates strong and accurate understanding of {topic} principles."
        elif score >= 5.0:
            summary = f"Demonstrates partial understanding of {topic}, with key details omitted."
        elif score >= 2.5:
            summary = f"The response shows foundational misunderstandings regarding {topic} in Java."
        else:
            summary = f"The response does not provide sufficient evidence to assess {topic}."

    # 2. Strengths (max 2 items, 15 words each)
    strengths = clamp_list_items(a4.strengths, max_items=2, max_words_per_item=15)

    # 3. Improvements (max 2 items, 15 words each)
    improvements = clamp_list_items(a4.areas_to_improve, max_items=2, max_words_per_item=15)

    # 4. Misconceptions (max 2 items, 15 words each, empty if none or high score)
    misconceptions = []
    if score < 8.5 and a4.conceptual_issues:
        misconceptions = clamp_list_items(a4.conceptual_issues, max_items=2, max_words_per_item=15)

    # 5. Personalized Feedback (max 60 words, 2-4 sentences, no summary duplication)
    raw_fb = deduplicate_summary_and_feedback(summary, a4.personalized_feedback)
    feedback = clamp_words_by_sentence(raw_fb, max_words=60)
    if not feedback:
        if score >= 8.5:
            feedback = f"You demonstrated accurate technical understanding of {topic}. Your grasp of fundamentals is sound, and you can now proceed to explore advanced design patterns."
        elif score >= 5.0:
            feedback = f"You identified several key elements of {topic}. Clarifying the remaining mechanisms and practicing with concrete code will make your technical explanation complete."
        else:
            feedback = f"Your response showed conceptual confusion regarding {topic}. Review the core rules and language constructs in Java to build a solid foundation."

    # 6. Actionable recommendations (max 2 items, max 20 words each)
    rec_actions = []
    for item in a4.practice_suggestions:
        if item and item not in rec_actions:
            rec_actions.append(item)
    if a4.next_learning_step and a4.next_learning_step not in rec_actions:
        rec_actions.append(a4.next_learning_step)
    if a4.learning_recommendation and a4.learning_recommendation not in rec_actions:
        rec_actions.append(a4.learning_recommendation)
    rec_actions = clamp_list_items(rec_actions, max_items=2, max_words_per_item=20)

    rec_single = clamp_words_by_sentence(a4.learning_recommendation, max_words=20)
    if not rec_single and rec_actions:
        rec_single = rec_actions[0]

    next_step = clamp_words_by_sentence(a4.next_learning_step, max_words=20)
    if not next_step and len(rec_actions) > 1:
        next_step = rec_actions[1]

    return {
        "overall_summary": summary,
        "strengths": strengths,
        "improvements": improvements,
        "areas_to_improve": improvements,
        "misconceptions": misconceptions,
        "feedback": feedback,
        "personalized_feedback": feedback,
        "recommendation": rec_single,
        "next_step": next_step,
        "practice_suggestions": rec_actions,
        "recommended_actions": rec_actions,
    }

def _agent_1_analyzer(provider: LLMProvider, context: Dict[str, Any]) -> Agent1Analysis:
    prompt = f"""
    You are Agent 1: Response Analyzer for an adaptive Java assessment system for MCA students.
    Task: Objectively identify the student's actual conceptual understanding based on their responses.
    DO NOT change or evaluate the predicted score. The score ({context['score']}/10) and learner level ({context['learner_level']}) are authoritative.

    Context:
    - Topic: {context['topic']}
    - Question 1: {context['question']}
    - Student Response 1: {context['initial_response']}
    - Question 2: {context['followup_question']}
    - Student Response 2: {context['followup_response']}
    - Authoritative Score: {context['score']}/10
    - Learner Level: {context['learner_level']}
    - Rubric Strengths: {context['strengths']}
    - Rubric Missing Concepts: {context['missing_concepts']}

    Strict Output Rules:
    1. Strengths: Maximum 2 concise items (5-15 words each). What did the student understand correctly? Return [] if empty/irrelevant response.
    2. Knowledge Gaps: Maximum 2 concise items (5-15 words each) identifying missing knowledge.
    3. Misconceptions: Maximum 2 items (5-15 words each), ONLY when the student explicitly stated a false concept. Do NOT invent misconceptions. Return [] if none.
    4. Missing Concepts: Maximum 2 items (5-15 words each) naming essential omitted concepts. Return [] if none.
    5. No repetition across categories. Each item must be a short sentence or phrase (5-15 words).
    """
    ans = provider.generate_structured_content(prompt, Agent1Analysis)
    if not ans.knowledge_gaps and ans.weaknesses:
        ans.knowledge_gaps = ans.weaknesses
    elif not ans.weaknesses and ans.knowledge_gaps:
        ans.weaknesses = ans.knowledge_gaps
    return ans

def _agent_2_feedback(provider: LLMProvider, context: Dict[str, Any], a1_out: Agent1Analysis) -> Agent2Feedback:
    prompt = f"""
    You are Agent 2: Personalized Feedback Agent for an adaptive Java assessment system for MCA students.
    Task: Generate direct, constructive educational feedback based on Agent 1's analysis and assessment evidence.

    Context:
    - Topic: {context['topic']}
    - Question 1: {context['question']}
    - Student Response 1: {context['initial_response']}
    - Question 2: {context['followup_question']}
    - Student Response 2: {context['followup_response']}
    - Authoritative Score: {context['score']}/10
    - Learner Level: {context['learner_level']}

    Agent 1 Analysis:
    {a1_out.model_dump_json(indent=2)}

    Strict Rules:
    1. Length: The explanation MUST be 2 to 4 sentences, MAXIMUM 60 WORDS.
    2. Content:
       - Mention what the student understood correctly.
       - Identify the most important missing concept or mistake, if one exists.
       - Give a clear, useful conceptual explanation or correction.
    3. Performance Tailoring:
       - Strong answer (score >= 8.5): Acknowledge understanding briefly. Identify at most one meaningful refinement. Do not invent weaknesses.
       - Partially correct (5.0 <= score < 8.5): Note what is correct, identify the primary missing concept, and explain it clearly.
       - Incorrect answer (score < 5.0 with misconception): Respectfully correct the misunderstanding in simple language.
       - Irrelevant/Empty answer (score < 2.5 or empty): State that the response did not provide enough relevant evidence to assess the concept; ask student to review and reattempt.
    4. Tone:
       - Zero excessive flattery (never say "exceptionally well-crafted", "fantastic grasp", "under the hood", "stellar job").
       - Direct, clear educational language appropriate for an MCA student learning programming.
       - Do NOT restate the entire original student answer.
    """
    return provider.generate_structured_content(prompt, Agent2Feedback)

def _agent_3_recommendation(provider: LLMProvider, context: Dict[str, Any], a1_out: Agent1Analysis, a2_out: Agent2Feedback) -> Agent3Recommendation:
    prompt = f"""
    You are Agent 3: Learning Recommendation Agent for an adaptive Java assessment system for MCA students.
    Task: Provide concrete, actionable next learning steps.

    Context:
    - Topic: {context['topic']}
    - Authoritative Score: {context['score']}/10
    - Learner Level: {context['learner_level']}

    Agent 1 Analysis: {a1_out.model_dump_json()}
    Agent 2 Feedback: {a2_out.model_dump_json()}

    Strict Rules:
    1. Provide MAXIMUM 2 recommendations / practice suggestions.
    2. Each recommendation MUST be ONE sentence of NO MORE THAN 20 WORDS.
    3. Must identify a concrete action (e.g., "Review method overriding syntax and write a subclass extending a parent class.").
    4. Avoid vague suggestions such as "study more", "keep learning", or "explore advanced topics".
    5. Align with student performance:
       - Strong: Appropriate next advanced topic or practical implementation exercise.
       - Partial: Targeted exercise focusing on the missing concept.
       - Incorrect: Foundational exercise reviewing the basic definition and syntax.
       - Irrelevant/Empty: Review core topic materials and reattempt the assessment.
    """
    return provider.generate_structured_content(prompt, Agent3Recommendation)

def _agent_4_coordinator(provider: LLMProvider, context: Dict[str, Any], a1_out: Agent1Analysis, a2_out: Agent2Feedback, a3_out: Agent3Recommendation) -> Agent4Coordinator:
    prompt = f"""
    You are Agent 4: Feedback Coordinator for an adaptive Java assessment system for MCA students.
    Task: Synthesize the outputs of Agents 1-3 into one concise, consistent final feedback object.

    Context:
    - Authoritative Score: {context['score']}/10
    - Learner Level: {context['learner_level']}
    - Topic: {context['topic']}

    Agent 1 (Analysis): {a1_out.model_dump_json()}
    Agent 2 (Feedback): {a2_out.model_dump_json()}
    Agent 3 (Recommendation): {a3_out.model_dump_json()}

    Strict Synthesis Rules:
    1. Feedback Summary (overall_summary): Exactly ONE sentence, MAXIMUM 20 WORDS.
    2. Strengths: Maximum 2 concise items (5-15 words each). If empty/irrelevant response, return [].
    3. Areas for Improvement (areas_to_improve): Maximum 2 concise items (5-15 words each).
    4. Conceptual Issues (conceptual_issues): Maximum 2 items (5-15 words each), ONLY when actual misconceptions were detected. Return [] otherwise.
    5. Personalized Feedback (personalized_feedback): Strictly 2-4 sentences, MAXIMUM 60 WORDS. Direct, educational explanation. DO NOT repeat overall_summary verbatim!
    6. Learning Recommendations (practice_suggestions & learning_recommendation): Maximum 2 actionable recommendations, each ONE sentence of MAXIMUM 20 WORDS.
    7. Overall visibility: Total feedback must be concise, around 80-120 words across all fields.
    8. Zero flattery, zero filler, zero duplicate statements.
    """
    return provider.generate_structured_content(prompt, Agent4Coordinator)


def run_multi_agent_feedback(
    question: str,
    initial_response: str,
    followup_question: str,
    followup_response: str,
    topic: str,
    auth_assessment: Dict[str, Any],
    progress_callback=None
) -> Dict[str, Any]:
    """
    Executes the 4-agent LLM pipeline to generate personalized feedback.
    Applies strict output validation and normalization before returning.
    Raises exception if API fails, which will trigger the robust rule-based fallback.
    """
    provider = LLMProvider()
    
    context = {
        "question": question,
        "initial_response": initial_response,
        "followup_question": followup_question,
        "followup_response": followup_response,
        "score": auth_assessment["score"],
        "learner_level": auth_assessment["learner_level"],
        "strengths": auth_assessment.get("strengths", []),
        "missing_concepts": auth_assessment.get("missing_concepts", []),
        "topic": topic
    }
    
    logger.info("Starting Multi-Agent LLM pipeline...")
    total_start = time.time()
    latencies = {}
    
    if progress_callback:
        progress_callback("Generating personalized feedback...\n\nRunning Response Analyzer...")
        
    t0 = time.time()
    a1_out = _agent_1_analyzer(provider, context)
    latencies['agent_1_sec'] = round(time.time() - t0, 2)
    logger.info(f"Agent 1 (Response Analyzer) complete in {latencies['agent_1_sec']}s.")
    
    if progress_callback:
        progress_callback("Generating personalized feedback...\n\n✓ Response Analyzer\n⏳ Personalized Feedback")
        
    t0 = time.time()
    a2_out = _agent_2_feedback(provider, context, a1_out)
    latencies['agent_2_sec'] = round(time.time() - t0, 2)
    logger.info(f"Agent 2 (Personalized Feedback) complete in {latencies['agent_2_sec']}s.")
    
    if progress_callback:
        progress_callback("Generating personalized feedback...\n\n✓ Response Analyzer\n✓ Personalized Feedback\n⏳ Learning Recommendation")
        
    t0 = time.time()
    a3_out = _agent_3_recommendation(provider, context, a1_out, a2_out)
    latencies['agent_3_sec'] = round(time.time() - t0, 2)
    logger.info(f"Agent 3 (Learning Recommendation) complete in {latencies['agent_3_sec']}s.")
    
    if progress_callback:
        progress_callback("Generating personalized feedback...\n\n✓ Response Analyzer\n✓ Personalized Feedback\n✓ Learning Recommendation\n⏳ Coordinator")
        
    t0 = time.time()
    a4_out = _agent_4_coordinator(provider, context, a1_out, a2_out, a3_out)
    latencies['agent_4_sec'] = round(time.time() - t0, 2)
    logger.info(f"Agent 4 (Coordinator) complete in {latencies['agent_4_sec']}s.")
    
    if progress_callback:
        progress_callback("Generating personalized feedback...\n\n✓ Response Analyzer\n✓ Personalized Feedback\n✓ Learning Recommendation\n✓ Coordinator")
        
    total_time = round(time.time() - total_start, 2)
    latencies['total_llm_sec'] = total_time
    latencies['model_used'] = LLMProvider._working_model or provider.model
    latencies['total_requests'] = LLMProvider._total_requests
    latencies['total_retries'] = LLMProvider._total_retries
    
    # Apply strict validation and normalization
    validated_output = normalize_and_validate_feedback(a4_out, context)
    validated_output["mode"] = "Multi-Agent LLM — Gemini"
    validated_output["feedback_mode"] = "Multi-Agent LLM — Gemini"
    validated_output["working_model"] = latencies['model_used']
    validated_output["latencies"] = latencies
    validated_output["debug_trace"] = {
        "agent_1": a1_out.model_dump(),
        "agent_2": a2_out.model_dump(),
        "agent_3": a3_out.model_dump(),
        "agent_4": a4_out.model_dump(),
    }
    
    return validated_output
