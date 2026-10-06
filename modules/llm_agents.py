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
    understanding: str = Field(description="Analysis of conceptual understanding")
    strengths: List[str] = Field(description="List of conceptual strengths")
    weaknesses: List[str] = Field(description="List of conceptual weaknesses")
    misconceptions: List[str] = Field(description="Identified misconceptions")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")

class Agent2Feedback(BaseModel):
    feedback_summary: str = Field(description="Summary of personalized feedback")
    strengths: List[str] = Field(description="Strengths to highlight to the student")
    areas_to_improve: List[str] = Field(description="Areas the student needs to improve")
    explanation: str = Field(description="Personalized explanation clarifying the concept")
    encouragement: str = Field(description="Encouraging closing statement")

class Agent3Recommendation(BaseModel):
    recommended_topic: str = Field(description="Topic for next learning step")
    reason: str = Field(description="Why this topic is recommended")
    recommended_action: str = Field(description="Specific actionable advice")
    practice_suggestions: List[str] = Field(description="List of specific practice exercises")
    next_learning_step: str = Field(description="The immediate next step to take")

class Agent4Coordinator(BaseModel):
    overall_summary: str = Field(description="Cohesive summary of feedback")
    strengths: List[str] = Field(description="Synthesized strengths")
    areas_to_improve: List[str] = Field(description="Synthesized areas to improve")
    conceptual_issues: List[str] = Field(description="Synthesized conceptual issues or misconceptions")
    personalized_feedback: str = Field(description="Final cohesive personalized explanation")
    learning_recommendation: str = Field(description="Synthesized learning recommendation")
    next_learning_step: str = Field(description="Final actionable next step")
    practice_suggestions: List[str] = Field(default=[], description="Final synthesized practice suggestions")

class LLMProvider:
    """Clean abstraction for LLM API providers."""
    
    _last_request_time = 0.0
    _total_requests = 0
    _total_retries = 0
    _working_model = None

    def __init__(self):
        self.provider = os.environ.get("LLM_PROVIDER", "gemini").lower()
        self.api_key = os.environ.get("LLM_API_KEY", "")
        self.model = os.environ.get("LLM_MODEL", "gemini-3.5-flash-lite")
        
        if not self.api_key:
            raise ValueError("LLM_API_KEY environment variable is not set.")
            
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


def _agent_1_analyzer(provider: LLMProvider, context: Dict[str, Any]) -> Agent1Analysis:
    prompt = f"""
    You are Agent 1: Response Analyzer for a Java assessment framework.
    Task: Analyze the student's conceptual understanding based on their responses.
    DO NOT change or evaluate the predicted score. The score is authoritative.
    
    - Topic: {context['topic']}
    - Question 1: {context['question']}
    - Student Response 1: {context['initial_response']}
    - Question 2: {context['followup_question']}
    - Student Response 2: {context['followup_response']}
    - Authoritative Score: {context['score']}/10
    - Learner Level: {context['learner_level']}
    - Detected Strengths: {context['strengths']}
    - Missing Concepts: {context['missing_concepts']}
    
    Provide your structured analysis.
    """
    return provider.generate_structured_content(prompt, Agent1Analysis)

def _agent_2_feedback(provider: LLMProvider, context: Dict[str, Any], a1_out: Agent1Analysis) -> Agent2Feedback:
    prompt = f"""
    You are Agent 2: Personalized Feedback Agent.
    Task: Generate highly specific feedback for this student.
    
    Context:
    - Topic: {context['topic']}
    - Original Question: {context['question']}
    - Student Response: {context['initial_response']}
    - Authoritative Score: {context['score']}/10
    - Learner Level: {context['learner_level']}
    
    Agent 1 Analysis:
    {a1_out.model_dump_json(indent=2)}
    
    If score is high, reinforce and push further. If medium, target gaps. If low, correct misconceptions simply.
    Provide specific, actionable feedback explaining exactly where their reasoning succeeded or failed.
    """
    return provider.generate_structured_content(prompt, Agent2Feedback)

def _agent_3_recommendation(provider: LLMProvider, context: Dict[str, Any], a1_out: Agent1Analysis, a2_out: Agent2Feedback) -> Agent3Recommendation:
    prompt = f"""
    You are Agent 3: Learning Recommendation Agent.
    Task: Generate an actionable learning plan based on detected weaknesses.
    
    Context:
    - Topic: {context['topic']}
    - Authoritative Score: {context['score']}/10
    - Learner Level: {context['learner_level']}
    
    Agent 1 Analysis: {a1_out.model_dump_json()}
    Agent 2 Feedback: {a2_out.model_dump_json()}
    
    Recommend a specific next topic, specific action, and 2-3 specific practice suggestions.
    """
    return provider.generate_structured_content(prompt, Agent3Recommendation)

def _agent_4_coordinator(provider: LLMProvider, context: Dict[str, Any], a1_out: Agent1Analysis, a2_out: Agent2Feedback, a3_out: Agent3Recommendation) -> Agent4Coordinator:
    prompt = f"""
    You are Agent 4: Feedback Coordinator.
    Task: Produce one final, cohesive, and perfectly structured response synthesizing the outputs of Agents 1-3.
    DO NOT introduce new claims unsupported by the previous agents or student responses.
    
    Context:
    - Authoritative Score: {context['score']}/10
    - Learner Level: {context['learner_level']}
    
    Agent 1 (Analysis): {a1_out.model_dump_json()}
    Agent 2 (Feedback): {a2_out.model_dump_json()}
    Agent 3 (Recommendation): {a3_out.model_dump_json()}
    
    Synthesize this into the final response format.
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
    
    if progress_callback:
        progress_callback("Generating personalized feedback...\n\n✓ Response Analyzer\n✓ Personalized Feedback\n✓ Learning Recommendation\n✓ Coordinator")
        
    total_time = round(time.time() - total_start, 2)
    latencies['total_llm_sec'] = total_time
    latencies['model_used'] = LLMProvider._working_model or provider.model
    latencies['total_requests'] = LLMProvider._total_requests
    latencies['total_retries'] = LLMProvider._total_retries
    
    logger.info(f"Agent 4 (Coordinator) complete in {latencies['agent_4_sec']}s. Total time: {total_time}s.")
    
    # Store complete LLM trace in context (useful for logging later)
    return {
        "overall_summary": a4_out.overall_summary,
        "strengths": a4_out.strengths,
        "improvements": a4_out.areas_to_improve,
        "misconceptions": a4_out.conceptual_issues,
        "feedback": a4_out.personalized_feedback,
        "recommendation": a4_out.learning_recommendation,
        "practice_suggestions": a4_out.practice_suggestions,
        "next_step": a4_out.next_learning_step,
        "mode": "Multi-Agent LLM",
        "working_model": latencies['model_used'],
        "latencies": latencies,
        "debug_trace": {
            "agent_1": a1_out.model_dump(),
            "agent_2": a2_out.model_dump(),
            "agent_3": a3_out.model_dump(),
            "agent_4": a4_out.model_dump(),
        }
    }
