"""
Feedback module for JCAB.
Handles rule-based preliminary feedback and provides a stub for future LLM integration.
"""
from typing import Any, Dict

def generate_development_fallback_feedback(score: float, learner_level: Any) -> Dict[str, Any]:
    """
    MODE 1 - DEVELOPMENT FALLBACK
    Rule-based feedback using preliminary model output.
    Clearly marked internally   as fallback/development logic.
    """
    if score >= 8:
        strengths = [
            "Strong conceptual understanding.",
            "The response addresses the assessed concept effectively.",
            "The student demonstrates consistency across the conversation."
        ]
        improvements = [
            "Add deeper examples or edge cases to strengthen the explanation."
        ]
        feedback = (
            "The response demonstrates strong understanding of the assessed Java concept."
        )
        recommendation = (
            "Proceed to advanced Java concepts such as polymorphism, abstraction, interfaces and design patterns."
        )
    elif score >= 6:
        strengths = [
            "Reasonable understanding of the core concept.",
            "The student attempts to address the follow-up question."
        ]
        improvements = [
            "Some conceptual details could be explained more precisely.",
            "The explanation would benefit from concrete examples."
        ]
        feedback = (
            "The response demonstrates a reasonable understanding of the concept, although some areas need greater clarity."
        )
        recommendation = (
            "Review the core concept and practice explaining it using simple Java examples."
        )
    elif score >= 4:
        strengths = [
            "The student identifies some relevant aspects of the concept.",
            "The student attempts both parts of the assessment."
        ]
        improvements = [
            "Important conceptual details are missing.",
            "The explanation requires greater clarity and accuracy."
        ]
        feedback = (
            "The response demonstrates partial understanding. Several important aspects require further clarification."
        )
        recommendation = (
            "Review the fundamentals and practice answering conceptual Java questions step by step."
        )
    else:
        strengths = [
            "The student attempted the assessment."
        ]
        improvements = [
            "The core concept is not sufficiently explained.",
            "Important Java terminology or relationships are missing."
        ]
        feedback = (
            "The response indicates that the fundamental concept requires further study."
        )
        recommendation = (
            "Revise the fundamentals before moving to advanced concepts."
        )

    return {
        "strengths": strengths,
        "improvements": improvements,
        "feedback": feedback,
        "recommendation": recommendation,
        "mode": "Development Fallback (Rule-Based)"
    }


def generate_llm_feedback(
    question: str,
    initial_response: str,
    followup_question: str,
    followup_response: str,
    score: float,
    learner_level: Any,
    topic: str,
    progress_callback=None
) -> Dict[str, Any]:
    """
    MODE 2 - LLM FEEDBACK (Multi-Agent Integration)
    """
    from modules.llm_agents import run_multi_agent_feedback
    return run_multi_agent_feedback(
        question, initial_response, followup_question, followup_response, score, learner_level, topic, progress_callback
    )

def get_feedback(
    question: str,
    initial_response: str,
    followup_question: str,
    followup_response: str,
    score: float,
    learner_level: Any,
    topic: str,
    use_llm: bool = False,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Returns feedback based on the selected mode.
    Falls back to development logic if LLM mode fails or is disabled.
    """
    if use_llm:
        try:
            return generate_llm_feedback(
                question, initial_response, followup_question, followup_response, score, learner_level, topic, progress_callback
            )
        except Exception as e:
            import os
            import logging
            logging.getLogger("JCAB").error(f"LLM Multi-Agent Pipeline failed: {e}. Falling back to rule-based.")
            if os.environ.get("LLM_REQUIRED_FOR_TEST", "false").lower() == "true":
                raise e
            fallback = generate_development_fallback_feedback(score, learner_level)
            fallback["fallback_reason"] = str(e)
            return fallback
            
    fallback = generate_development_fallback_feedback(score, learner_level)
    fallback["fallback_reason"] = "LLM configuration disabled"
    return fallback
