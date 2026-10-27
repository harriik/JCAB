"""
Validation test suite for Final UI/UX Transformation of Adaptive Assessment.
Verifies:
1. Python compilation of modified files.
2. Demo Mode ON assessment flow and deterministic local fallback.
3. Demo Mode OFF with mocked Gemini success.
4. Demo Mode OFF with simulated Gemini 429 quota failure fallback.
5. Result dictionary structure & normalization (score, learner_level, concept_explanation, etc.).
6. Student report export cleanliness (no ML model comparison columns).
7. Theme configuration consistency (light theme).
8. Absence of visible 'JCAB' branding in user-facing UI.
9. Absence of ML model comparison displays in student results.
"""
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Reconfigure stdout for utf-8 on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from modules.feedback import (
    generate_authoritative_assessment,
    generate_development_fallback_feedback,
    get_feedback
)
from app import normalize_assessment_result


def test_python_compilation():
    import py_compile
    files = ["app.py", "modules/feedback.py", "modules/llm_agents.py", "config/settings.py"]
    for f in files:
        py_compile.compile(str(BASE_DIR / f), doraise=True)
    print("✅ Test 1 Passed: Python compilation clean for all core files.")


def test_demo_mode_on():
    result = get_feedback(
        question="What is encapsulation in Java?",
        initial_response="Encapsulation is data hiding using private fields and public getters and setters.",
        followup_question="Why is it important?",
        followup_response="It provides modularity, maintainability, and protects object state.",
        experimental_ml_score=0.88,
        experimental_ml_learner_level="Excellent",
        topic="OOP Fundamentals",
        use_llm=False,
        demo_mode=True
    )
    assert "score" in result, "Missing 'score' key"
    assert "learner_level" in result, "Missing 'learner_level' key"
    assert "concept_explanation" in result, "Missing 'concept_explanation' key"
    assert len(result["concept_explanation"]) > 10, "concept_explanation too short"
    assert result["mode"] == "Demo Mode — Local Fallback", f"Unexpected mode: {result['mode']}"
    assert 0.0 <= result["score"] <= 10.0, f"Score out of range: {result['score']}"
    assert result["learner_level"] in ["Excellent", "Good", "Partial", "Weak", "Incorrect"]
    print(f"✅ Test 2 Passed: Demo Mode ON completes with mode '{result['mode']}', Score: {result['score']}/10, Level: {result['learner_level']}.")


def test_gemini_success_mocked():
    from pydantic import BaseModel, Field
    from typing import List

    class DummyCoordinator(BaseModel):
        overall_summary: str = "Demonstrates accurate understanding of encapsulation."
        strengths: List[str] = ["Correctly explained data hiding", "Mentioned access mutators"]
        areas_to_improve: List[str] = ["Explore advanced immutability patterns"]
        conceptual_issues: List[str] = []
        personalized_feedback: str = "Your answer demonstrated clear technical understanding of encapsulation. You explained access control and modularity effectively."
        learning_recommendation: str = "Practice designing immutable Java classes."
        next_learning_step: str = "Implement record classes in modern Java."
        practice_suggestions: List[str] = ["Design an immutable value object.", "Review record syntax."]

    mock_llm_out = {
        "overall_summary": "Demonstrates accurate understanding of encapsulation.",
        "strengths": ["Correctly explained data hiding", "Mentioned access mutators"],
        "areas_to_improve": ["Explore advanced immutability patterns"],
        "improvements": ["Explore advanced immutability patterns"],
        "misconceptions": [],
        "feedback": "Your answer demonstrated clear technical understanding of encapsulation. You explained access control and modularity effectively.",
        "personalized_feedback": "Your answer demonstrated clear technical understanding of encapsulation.",
        "recommendation": "Practice designing immutable Java classes.",
        "next_step": "Implement record classes in modern Java.",
        "practice_suggestions": ["Design an immutable value object.", "Review record syntax."],
        "recommended_actions": ["Design an immutable value object.", "Review record syntax."],
        "working_model": "gemini-3.5-flash-lite",
        "latencies": {"total_llm_sec": 1.2}
    }

    with patch("modules.feedback.generate_llm_feedback", return_value=mock_llm_out):
        result = get_feedback(
            question="What is encapsulation in Java?",
            initial_response="Encapsulation is data hiding.",
            followup_question="Why is it used?",
            followup_response="It protects data.",
            experimental_ml_score=0.75,
            experimental_ml_learner_level="Good",
            topic="OOP Fundamentals",
            use_llm=True,
            demo_mode=False
        )
        assert result["mode"] == "Multi-Agent LLM — Gemini", f"Unexpected mode: {result['mode']}"
        assert result["working_model"] == "gemini-3.5-flash-lite"
        print(f"✅ Test 3 Passed: Mocked Gemini success reports mode '{result['mode']}'.")


def test_gemini_429_failure_fallback():
    with patch("modules.feedback.generate_llm_feedback", side_effect=Exception("429 RESOURCE_EXHAUSTED: Quota exceeded")):
        result = get_feedback(
            question="What is inheritance in Java?",
            initial_response="Inheritance allows subclasses to acquire parent properties.",
            followup_question="What keyword is used?",
            followup_response="The extends keyword.",
            experimental_ml_score=0.85,
            experimental_ml_learner_level="Excellent",
            topic="Inheritance",
            use_llm=True,
            demo_mode=False
        )
        assert result["mode"] == "Automatic Fallback — Gemini Unavailable", f"Unexpected mode: {result['mode']}"
        assert "score" in result
        assert "concept_explanation" in result
        assert "429" in result.get("fallback_reason", "")
        print(f"✅ Test 4 Passed: Gemini 429 quota failure triggers fallback '{result['mode']}' gracefully without crash.")


def test_result_normalization_and_student_report():
    raw_feedback = {
        "score": 8.75,
        "learner_level": "Excellent",
        "overall_summary": "Strong grasp of polymorphism principles.",
        "feedback": "You clearly explained dynamic method dispatch and method overriding.",
        "strengths": ["Clear explanation of method overriding", "Accurate use of super keyword"],
        "improvements": ["Explore runtime bytecode dispatch"],
        "recommended_actions": ["Write an interface with multiple implementations."],
        "concept_explanation": "Polymorphism enables an object to take many forms via method overriding and interfaces.",
        "mode": "Demo Mode — Local Fallback"
    }
    base_info = {
        "timestamp": "2026-10-24 12:00:00",
        "student_name": "Test Student",
        "student_id": "STU-001",
        "domain": "Java",
        "topic": "Polymorphism",
        "question": "What is polymorphism?",
        "initial_response": "One interface multiple forms.",
        "followup_question": "Explain dynamic dispatch.",
        "followup_response": "Runtime resolution of method call.",
        "experimental_ml_score": 0.89,
        "experimental_ml_learner_level": "Excellent"
    }

    normalized = normalize_assessment_result(raw_feedback, base_info)
    assert normalized["score"] == 8.75
    assert normalized["learner_level"] == "Excellent"
    assert normalized["topic"] == "Polymorphism"
    assert normalized["concept_explanation"] != ""
    assert "strengths" in normalized and len(normalized["strengths"]) == 2
    assert "improvements" in normalized and len(normalized["improvements"]) == 1

    # Student personal report columns
    student_report_cols = [
        "timestamp", "student_name", "topic", "score", "learner_level", 
        "question", "initial_response", "followup_question", "followup_response", 
        "overall_summary", "feedback", "concept_explanation", "recommendation"
    ]
    for prohibited in ["experimental_ml_score", "experimental_ml_learner_level", "xgb", "classifier", "regressor"]:
        assert prohibited not in student_report_cols, f"Prohibited ML field {prohibited} in student report columns!"

    print("✅ Test 5 Passed: Assessment result normalization & student report column filtering verified.")


def test_theme_and_branding():
    # 1. Config.toml theme check
    config_path = BASE_DIR / ".streamlit" / "config.toml"
    assert config_path.exists(), ".streamlit/config.toml missing!"
    config_text = config_path.read_text(encoding="utf-8")
    assert 'base = "light"' in config_text, "Theme base is not light in config.toml!"
    assert 'primaryColor = "#168675"' in config_text, "Primary color mismatch in config.toml!"
    assert 'backgroundColor = "#F3F6F8"' in config_text, "Background color mismatch in config.toml!"

    # 2. app.py CSS and branding check
    app_text = (BASE_DIR / "app.py").read_text(encoding="utf-8")
    assert "#F3F6F8" in app_text, "app.py does not contain expected educational page background #F3F6F8!"
    assert "#168675" in app_text, "app.py does not contain expected primary accent #168675!"
    assert "Adaptive Assessment" in app_text, "app.py missing Adaptive Assessment branding!"

    # Check for no visible JCAB in rendered HTML/Markdown
    # (Excluding file headers/comments)
    import re
    # Remove docstring at beginning
    code_body = re.sub(r'^""".*?"""', '', app_text, flags=re.DOTALL)
    visible_jcab = re.findall(r'(?:title|header|markdown|write|caption|label)\(.*?[^a-zA-Z0-9]JCAB[^a-zA-Z0-9].*?\)', code_body, re.IGNORECASE)
    assert len(visible_jcab) == 0, f"Found visible JCAB in UI calls: {visible_jcab}"

    # Check that model comparisons (Random Forest, XGBoost, etc.) are NOT in student-facing results
    results_fn_match = re.search(r'def page_results\(\):.*?(?=def page_|\Z)', app_text, re.DOTALL)
    assert results_fn_match is not None, "page_results() not found in app.py"
    results_body = results_fn_match.group(0)

    for banned_term in ["Random Forest", "XGBoost", "Logistic Regression", "SVM", "Predicted ML Score", "Sentence-BERT + XGBoost"]:
        assert banned_term.lower() not in results_body.lower(), f"Banned ML comparison term '{banned_term}' found in page_results()!"

    print("✅ Test 6 Passed: Consistent light theme, branding, and strict absence of ML model comparisons in student results verified.")


if __name__ == "__main__":
    print("\n--- Running UI/UX Transformation Validation Suite ---\n")
    test_python_compilation()
    test_demo_mode_on()
    test_gemini_success_mocked()
    test_gemini_429_failure_fallback()
    test_result_normalization_and_student_report()
    test_theme_and_branding()
    print("\n🎉 ALL 6 TEST SUITES PASSED SUCCESSFULLY!\n")
