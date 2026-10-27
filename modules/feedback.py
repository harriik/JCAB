"""
Feedback module for JCAB.
Handles rule-based preliminary feedback and provides a stub for future LLM integration.
"""
from typing import Any, Dict

import json
from pathlib import Path
import numpy as np

def generate_authoritative_assessment(
    topic: str,
    question: str,
    initial_response: str,
    followup_question: str,
    followup_response: str
) -> Dict[str, Any]:
    """
    Generates the core, student-facing assessment score and learner level
    using deterministic semantic similarity and rubrics.
    """
    # Load rubrics
    rubric_path = Path("data/java_fallback_rubrics.json")
    if rubric_path.exists():
        with open(rubric_path, "r", encoding="utf-8") as f:
            rubrics = json.load(f)
    else:
        rubrics = {}
        
    topic_rubric = rubrics.get(topic, {
        "expected_concepts": [topic.lower(), "java"],
        "important_keywords": [topic.lower().split()[0]],
        "reference_explanation": f"This is about {topic} in Java."
    })
    
    # 1. Semantic Similarity using existing model
    from modules.embeddings import load_embedding_model, generate_embedding
    model = load_embedding_model()
    
    student_text = f"{initial_response} {followup_response}".lower()
    ref_text = topic_rubric["reference_explanation"].lower()
    
    try:
        from sklearn.metrics.pairwise import cosine_similarity
        student_emb = generate_embedding(student_text, model)
        ref_emb = generate_embedding(ref_text, model)
        import numpy as np
        student_emb = np.array(student_emb).reshape(1, -1)
        ref_emb = np.array(ref_emb).reshape(1, -1)
        
        sim = float(cosine_similarity(student_emb, ref_emb)[0][0])
        sim = max(0.0, sim)
    except Exception as e:
        print(f"DEBUG: Embedding failed: {e}")
        sim = 0.5 # fallback if embedding fails
        
    # Base semantic similarity and coverage
    expected_concepts = [c.lower() for c in topic_rubric.get("expected_concepts", [])]
    keywords = [kw.lower() for kw in topic_rubric.get("important_keywords", [])]
    all_targets = list(set(keywords + expected_concepts))
    
    found_kws = set()
    for kw in all_targets:
        if kw in student_text:
            found_kws.add(kw)
        elif len(kw) > 4:
            if kw[:4] in student_text:
                found_kws.add(kw)
                
    coverage = len(found_kws) / len(all_targets) if all_targets else 0.0
    is_empty = len(student_text.strip()) < 15
    
    if is_empty:
        raw_score = 0.0
    else:
        # Blended score metric: 70% sim, 30% coverage
        norm_sim = max(0.0, min(1.0, (sim - 0.2) / 0.7))
        norm_cov = max(0.0, min(1.0, coverage / 0.8))
        blend = (norm_sim * 0.7) + (norm_cov * 0.3)
        
        if blend < 0.3:
            raw_score = blend * 10.0 # 0 to 3 (Irrelevant)
        elif blend < 0.75:
            # Incorrect to very weak partial
            raw_score = 3.0 + ((blend - 0.3) / 0.45 * 2.0) # 3 to 5
        elif blend < 0.80:
            # Partial
            raw_score = 5.0 + ((blend - 0.75) / 0.05 * 2.0) # 5 to 7
        elif blend < 0.85:
            # Incomplete
            raw_score = 7.0 + ((blend - 0.80) / 0.05 * 1.5) # 7 to 8.5
        else:
            # Strong
            raw_score = 8.5 + ((blend - 0.85) / 0.15 * 1.5) # 8.5 to 10
            
    final_score = max(0.0, min(10.0, raw_score))
    
    # Learner Level
    if final_score >= 8.5:
        level = "Excellent"
    elif final_score >= 7.0:
        level = "Good"
    elif final_score >= 5.0:
        level = "Partial"
    elif final_score >= 2.5:
        level = "Weak"
    else:
        level = "Incorrect"
        
    expected = [c.lower() for c in topic_rubric["expected_concepts"]]
    detected = [c for c in expected if c in student_text]
    missing = [c for c in expected if c not in student_text]
    
    strengths = []
    if detected:
        strengths.append(f"Correctly identified key concepts: {', '.join(detected)}.")
        if final_score >= 7.0:
            strengths.append(f"Demonstrated a strong grasp of {topic} principles.")
    else:
        strengths.append(f"Attempted to define {topic}, but foundational concepts were absent.")
        
    improvements = []
    if missing:
        improvements.append(f"Missing essential concepts for {topic}: {', '.join(missing)}.")
        
    return {
        "assessment_mode": "Local Heuristic Assessment",
        "score": round(final_score, 2),
        "learner_level": level,
        "strengths": strengths,
        "missing_concepts": missing,
        "areas_to_improve": improvements,
        "concept_explanation": topic_rubric.get("reference_explanation", f"Key conceptual mechanisms and principles of {topic} in Java."),
        "debug_sim": round(sim, 3),
        "debug_cov": round(coverage, 3)
    }

def generate_development_fallback_feedback(
    topic: str,
    question: str,
    initial_response: str,
    followup_question: str,
    followup_response: str,
    auth_assessment: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    MODE 1 - DEVELOPMENT FALLBACK
    Generates concise, constructive, performance-tailored feedback
    using deterministic heuristic assessment, adhering to strict word and item limits:
    - Overall summary: 1 sentence, max 20 words
    - Strengths: max 2 items, 5-15 words each
    - Areas for improvement: max 2 items, 5-15 words each
    - Personalized feedback: 2-4 sentences, max 60 words
    - Recommended next steps: max 2 items, max 20 words each
    """
    if auth_assessment is None:
        auth_assessment = generate_authoritative_assessment(
            topic, question, initial_response, followup_question, followup_response
        )
        
    final_score = auth_assessment["score"]
    level = auth_assessment["learner_level"]
    missing = auth_assessment.get("missing_concepts", [])
    raw_strengths = auth_assessment.get("strengths", [])
    
    student_text = f"{initial_response} {followup_response}".strip()
    is_empty_or_irrelevant = (final_score < 2.5) or (len(student_text) < 15)
    
    misconceptions = []
    
    # 1. Performance-Tailored Generation
    if is_empty_or_irrelevant:
        overall_summary = f"The response does not provide sufficient evidence to assess understanding of {topic}."
        strengths = []
        improvements = [f"Review foundational principles of {topic} before reattempting."]
        feedback = (
            f"Your response did not provide enough relevant evidence to evaluate your knowledge of {topic}. "
            f"Please review the core principles and attempt the question again."
        )
        recommendation = f"Review the question carefully and reattempt with a conceptual explanation."
        next_step = f"Review the basic definition of {topic} in your study materials."
        recommended_actions = [
            f"Review the basic definition and key concepts of {topic}.",
            f"Reattempt the question with a complete explanation."
        ]
        
    elif final_score < 5.0:
        # Incorrect answer
        overall_summary = f"The response shows fundamental misunderstandings regarding {topic} in Java."
        strengths = [s for s in raw_strengths if "Correctly" in s][:1]
        improvements = [
            f"Clarify the core definition and syntax of {topic}.",
            f"Avoid confusing {topic} with unrelated programming concepts."
        ]
        misconceptions = [f"Misidentified core mechanisms of {topic} in Java."]
        focus_concept = missing[0] if missing else topic
        feedback = (
            f"Your explanation misidentified key concepts of {topic}. In Java, {topic} relies on specific "
            f"rules and language constructs. Review the basic definitions before moving to complex usage."
        )
        recommendation = f"Review foundational Java tutorials on {topic} and practice simple examples."
        next_step = f"Write a minimal Java program illustrating {focus_concept}."
        recommended_actions = [
            f"Review foundational documentation and syntax rules for {topic}.",
            f"Write a minimal Java program illustrating {focus_concept}."
        ]
        
    elif final_score < 8.5:
        # Partially correct answer
        overall_summary = f"Demonstrates partial understanding of {topic}, with key technical details missing."
        strengths = []
        for s in raw_strengths:
            if "Correctly" in s:
                strengths.append(s)
        if not strengths:
            strengths.append(f"Demonstrated basic conceptual familiarity with {topic}.")
        strengths = strengths[:2]
        
        improvements = []
        if missing:
            improvements.append(f"Explain key missing concepts: {', '.join(missing[:2])}.")
        improvements.append(f"Provide greater technical depth on {topic} mechanisms.")
        improvements = improvements[:2]
        
        gap_concept = missing[0] if missing else "core mechanisms"
        feedback = (
            f"You correctly identified several elements of {topic}. However, your explanation needs a clearer "
            f"description of {gap_concept} in Java. Adding specific technical details will complete your answer."
        )
        recommendation = f"Practise explaining {gap_concept} with a short code example."
        next_step = f"Review how {gap_concept} operates in standard Java libraries."
        recommended_actions = [
            f"Review the role of {gap_concept} in Java.",
            f"Write a short code example demonstrating {gap_concept}."
        ]
        
    else:
        # Strong answer (score >= 8.5)
        overall_summary = f"Demonstrates strong and accurate understanding of {topic} principles."
        strengths = [
            f"Accurately explained core principles of {topic}.",
            f"Clearly addressed both initial and follow-up questions."
        ]
        improvements = [
            f"Explore edge cases and advanced implementation patterns for {topic}."
        ]
        feedback = (
            f"You accurately explained {topic} and demonstrated sound technical knowledge. "
            f"Your understanding of the fundamentals is solid. You can now explore advanced implementations."
        )
        recommendation = f"Explore advanced design patterns and best practices for {topic}."
        next_step = f"Implement an advanced example combining {topic} with other Java features."
        recommended_actions = [
            f"Explore advanced design patterns and best practices for {topic}.",
            f"Analyze practical edge cases or performance considerations."
        ]

    return {
        "feedback_mode": "Development Fallback (Rule-Based)",
        "mode": "Development Fallback (Rule-Based)",
        "score": final_score,
        "learner_level": level,
        "overall_summary": overall_summary,
        "strengths": strengths[:2],
        "improvements": improvements[:2],
        "areas_to_improve": improvements[:2],
        "misconceptions": misconceptions[:2],
        "feedback": feedback,
        "personalized_feedback": feedback,
        "recommendation": recommendation,
        "next_step": next_step,
        "practice_suggestions": recommended_actions[:2],
        "recommended_actions": recommended_actions[:2],
        "debug_sim": auth_assessment.get("debug_sim", 0.0),
        "debug_cov": auth_assessment.get("debug_cov", 0.0),
    }


def generate_llm_feedback(
    question: str,
    initial_response: str,
    followup_question: str,
    followup_response: str,
    topic: str,
    auth_assessment: Dict[str, Any],
    progress_callback=None
) -> Dict[str, Any]:
    """
    MODE 2 - LLM FEEDBACK (Multi-Agent Integration)
    """
    from modules.llm_agents import run_multi_agent_feedback
    return run_multi_agent_feedback(
        question, initial_response, followup_question, followup_response, topic, auth_assessment, progress_callback
    )

def get_feedback(
    question: str,
    initial_response: str,
    followup_question: str,
    followup_response: str,
    experimental_ml_score: float,
    experimental_ml_learner_level: str,
    topic: str,
    use_llm: bool = False,
    demo_mode: bool = False,
    progress_callback=None
) -> Dict[str, Any]:
    """
    Returns a single unified authoritative assessment object containing the 
    authoritative score/level and the experimental ML values.
    """
    # 1. Always generate the authoritative assessment first
    auth_assessment = generate_authoritative_assessment(
        topic, question, initial_response, followup_question, followup_response
    )
    
    # Base unified result object
    unified_result = {
        "score": auth_assessment["score"],
        "learner_level": auth_assessment["learner_level"],
        "assessment_mode": auth_assessment["assessment_mode"],
        "strengths": auth_assessment["strengths"],
        "areas_to_improve": auth_assessment["areas_to_improve"],
        "concept_explanation": auth_assessment.get("concept_explanation", ""),
        "experimental_ml_score": experimental_ml_score,
        "experimental_ml_learner_level": experimental_ml_learner_level
    }

    if use_llm:
        try:
            llm_result = generate_llm_feedback(
                question, initial_response, followup_question, followup_response, topic, auth_assessment, progress_callback
            )
            llm_result["mode"] = "Multi-Agent LLM — Gemini"
            llm_result["feedback_mode"] = "Multi-Agent LLM — Gemini"
            unified_result.update(llm_result)
            return unified_result
        except Exception as e:
            import os
            import logging
            logging.getLogger("JCAB").error(f"LLM Multi-Agent Pipeline failed: {e}. Falling back to rule-based.")
            if os.environ.get("LLM_REQUIRED_FOR_TEST", "false").lower() == "true":
                raise e
            fallback = generate_development_fallback_feedback(
                topic, question, initial_response, followup_question, followup_response, auth_assessment
            )
            fallback["mode"] = "Automatic Fallback — Gemini Unavailable"
            fallback["feedback_mode"] = "Automatic Fallback — Gemini Unavailable"
            fallback["fallback_reason"] = str(e)
            unified_result.update(fallback)
            return unified_result
            
    fallback = generate_development_fallback_feedback(
        topic, question, initial_response, followup_question, followup_response, auth_assessment
    )
    target_mode = "Demo Mode — Local Fallback" if demo_mode else "Automatic Fallback — Gemini Unavailable"
    fallback["mode"] = target_mode
    fallback["feedback_mode"] = target_mode
    fallback["fallback_reason"] = "Demo Mode active" if demo_mode else "Gemini pipeline disabled or bypassed"
    unified_result.update(fallback)
    return unified_result
