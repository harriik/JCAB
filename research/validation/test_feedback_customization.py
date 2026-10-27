"""
Comprehensive Validation Test for Gemini Feedback Customization and Fallback
Tests all 6 controlled cases:
1. Strong conceptual answer
2. Partially correct answer
3. Incorrect answer with misconception
4. Irrelevant answer
5. Empty answer
6. Gemini quota / API failure (fallback)
"""

import sys
import os
import re
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from modules.feedback import get_feedback, generate_development_fallback_feedback, generate_authoritative_assessment
from modules.llm_agents import clamp_words_by_sentence, clamp_list_items, deduplicate_summary_and_feedback, sanitize_text

FORBIDDEN_FLATTERY = [
    "exceptionally well-crafted",
    "fantastic grasp of the fundamentals",
    "keep pushing your knowledge further",
    "under the hood",
    "outstanding response",
    "stellar job",
    "brilliant explanation",
]

def count_words(text: str) -> int:
    if not text:
        return 0
    return len(text.strip().split())

def check_no_forbidden_flattery(text: str) -> bool:
    lower = text.lower()
    for phrase in FORBIDDEN_FLATTERY:
        if phrase in lower:
            return False
    return True

def run_case_assertions(case_num: int, name: str, res: dict) -> dict:
    summary = res.get("overall_summary", "")
    feedback = res.get("feedback", res.get("personalized_feedback", ""))
    strengths = res.get("strengths", [])
    improvements = res.get("improvements", res.get("areas_to_improve", []))
    actions = res.get("recommended_actions", res.get("practice_suggestions", []))
    score = res.get("score")
    level = res.get("learner_level")

    summary_words = count_words(summary)
    feedback_words = count_words(feedback)
    strengths_count = len(strengths)
    improvements_count = len(improvements)
    actions_count = len(actions)

    strengths_words = [count_words(s) for s in strengths]
    improvements_words = [count_words(i) for i in improvements]
    actions_words = [count_words(a) for a in actions]

    total_visible_words = summary_words + feedback_words + sum(strengths_words) + sum(improvements_words) + sum(actions_words)

    # Verbatim repetition check
    norm_sum = re.sub(r'[^a-zA-Z0-9\s]', '', summary.lower()).strip()
    norm_fb = re.sub(r'[^a-zA-Z0-9\s]', '', feedback.lower()).strip()
    is_verbatim_duplicate = bool(norm_sum and norm_sum in norm_fb)

    # Flattery check
    all_text = f"{summary} {feedback} {' '.join(strengths)} {' '.join(improvements)} {' '.join(actions)}"
    has_flattery = not check_no_forbidden_flattery(all_text)

    # Verification checks
    passed_summary_limit = summary_words <= 20
    passed_feedback_limit = feedback_words <= 60
    passed_strengths_limit = strengths_count <= 2 and all(w <= 20 for w in strengths_words)
    passed_improvements_limit = improvements_count <= 2 and all(w <= 20 for w in improvements_words)
    passed_actions_limit = actions_count <= 2 and all(w <= 20 for w in actions_words)

    all_passed = (
        passed_summary_limit
        and passed_feedback_limit
        and passed_strengths_limit
        and passed_improvements_limit
        and passed_actions_limit
        and not is_verbatim_duplicate
        and not has_flattery
        and score is not None
        and level is not None
    )

    print(f"\n[{'PASS' if all_passed else 'FAIL'}] Case {case_num}: {name}")
    print(f"  Authoritative Score: {score}/10 | Level: {level}")
    print(f"  Assessment Summary ({summary_words} words, limit <= 20): \"{summary}\"")
    print(f"  Strengths ({strengths_count} items, limit <= 2): {strengths}")
    print(f"  Improvements ({improvements_count} items, limit <= 2): {improvements}")
    print(f"  Personalized Feedback ({feedback_words} words, limit <= 60): \"{feedback}\"")
    print(f"  Recommended Next Steps ({actions_count} items, limit <= 2): {actions}")
    print(f"  Total visible words: {total_visible_words} (ideal 80-120)")
    print(f"  Verbatim repetition: {'YES (FAIL)' if is_verbatim_duplicate else 'NO (PASS)'}")
    print(f"  Excessive flattery: {'FOUND (FAIL)' if has_flattery else 'NONE (PASS)'}")

    return {
        "case_num": case_num,
        "name": name,
        "passed": all_passed,
        "score": score,
        "level": level,
        "summary_words": summary_words,
        "feedback_words": feedback_words,
        "total_visible_words": total_visible_words,
        "strengths_count": strengths_count,
        "improvements_count": improvements_count,
        "actions_count": actions_count,
        "has_flattery": has_flattery,
        "is_verbatim_duplicate": is_verbatim_duplicate,
    }


def main():
    print("=" * 70)
    print("      FINAL CUSTOMIZATION VERIFICATION: 6 CONTROLLED CASES")
    print("=" * 70)

    results = []

    # Case 1: Strong conceptual answer
    topic_1 = "Inheritance"
    q_1 = "What is inheritance in Java and why is it used?"
    a_1 = "Inheritance allows a child class to inherit fields and methods from a parent class using the extends keyword. It promotes code reusability."
    fq_1 = "What is the difference between single and multilevel inheritance?"
    fa_1 = "Single inheritance has one subclass extending one parent class. Multilevel inheritance is a chain where a child class extends a class that is itself a subclass."
    res_1 = get_feedback(q_1, a_1, fq_1, fa_1, experimental_ml_score=9.0, experimental_ml_learner_level="Excellent", topic=topic_1, use_llm=False)
    results.append(run_case_assertions(1, "Strong Conceptual Answer (Inheritance)", res_1))

    # Case 2: Partially correct answer
    topic_2 = "Polymorphism"
    q_2 = "What is polymorphism in Java?"
    a_2 = "Polymorphism means many forms. In Java it means methods can have different implementations."
    fq_2 = "Can you explain method overriding vs method overloading?"
    fa_2 = "Overloading is when methods have the same name but different parameters. I don't remember method overriding."
    res_2 = get_feedback(q_2, a_2, fq_2, fa_2, experimental_ml_score=6.0, experimental_ml_learner_level="Partial", topic=topic_2, use_llm=False)
    results.append(run_case_assertions(2, "Partially Correct Answer (Polymorphism)", res_2))

    # Case 3: Incorrect answer with a misconception
    topic_3 = "Concurrency vs Parallelism"
    q_3 = "What is the difference between concurrency and parallelism in Java?"
    a_3 = "Concurrency is when variables are made private. Parallelism is inheriting from multiple interfaces."
    fq_3 = "How does the Fork/Join framework help achieve parallelism?"
    fa_3 = "Fork/Join joins strings and threads together to avoid syntax errors."
    res_3 = get_feedback(q_3, a_3, fq_3, fa_3, experimental_ml_score=3.0, experimental_ml_learner_level="Incorrect", topic=topic_3, use_llm=False)
    results.append(run_case_assertions(3, "Incorrect Answer with Misconception (Concurrency)", res_3))

    # Case 4: Irrelevant answer
    topic_4 = "OOP Fundamentals"
    q_4 = "Explain the four pillars of OOP in Java."
    a_4 = "I went to the store today and bought some pizza and cold drinks with my friends."
    fq_4 = "How does encapsulation protect data?"
    fa_4 = "Pizza is delicious especially with extra cheese."
    res_4 = get_feedback(q_4, a_4, fq_4, fa_4, experimental_ml_score=1.0, experimental_ml_learner_level="Incorrect", topic=topic_4, use_llm=False)
    results.append(run_case_assertions(4, "Irrelevant Answer (OOP Fundamentals)", res_4))

    # Case 5: Empty answer
    topic_5 = "Exception Handling"
    q_5 = "Explain the difference between checked and unchecked exceptions."
    a_5 = "idk"
    fq_5 = "What is the purpose of the finally block?"
    fa_5 = "none"
    res_5 = get_feedback(q_5, a_5, fq_5, fa_5, experimental_ml_score=0.0, experimental_ml_learner_level="Incorrect", topic=topic_5, use_llm=False)
    results.append(run_case_assertions(5, "Empty / Very Short Answer (Exception Handling)", res_5))

    # Case 6: Gemini quota / API failure (Fallback Triggered)
    print("\n--- Testing Case 6: Gemini quota / API failure simulation ---")
    class MockQuotaException(Exception):
        pass

    def mock_failing_llm(*args, **kwargs):
        raise MockQuotaException("429 RESOURCE_EXHAUSTED: GenerateRequestsPerDayPerProjectPerModel-FreeTier limit exceeded")

    import modules.feedback as fb
    orig_llm = fb.generate_llm_feedback
    fb.generate_llm_feedback = mock_failing_llm

    try:
        res_6 = get_feedback(
            question="What is encapsulation in Java?",
            initial_response="Encapsulation is hiding internal object data and restricting direct access using private fields.",
            followup_question="How do getters and setters work?",
            followup_response="Getters retrieve private fields and setters validate and update them.",
            experimental_ml_score=8.8,
            experimental_ml_learner_level="Excellent",
            topic="OOP Fundamentals",
            use_llm=True
        )
        results.append(run_case_assertions(6, "Gemini Quota / API Failure (Graceful Fallback)", res_6))
    finally:
        fb.generate_llm_feedback = orig_llm

    print("\n" + "=" * 70)
    print("SUMMARY OF VERIFICATION")
    print("=" * 70)
    all_passed = all(r["passed"] for r in results)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"Case {r['case_num']} ({r['name']}): {status} | Summary: {r['summary_words']}w, Feedback: {r['feedback_words']}w, Total: {r['total_visible_words']}w")
    print("=" * 70)
    print(f"OVERALL RESULT: {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'}")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
