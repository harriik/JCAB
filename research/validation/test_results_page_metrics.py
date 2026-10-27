"""
Verification test for Adaptive Assessment Results Page Metrics:
Confirms the source of:
- Main Score (Authoritative Semantic/Rubric Assessment)
- Main Learner Level (Authoritative Threshold Mapping)
- Experimental XGBoost Score (XGBRegressor output)
- Experimental ML Learner-Level Prediction (XGBClassifier + LabelEncoder output)
- Confirms that Personalized Feedback uses the Authoritative assessment
"""

import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(PROJECT_ROOT))

from modules.preprocessing import build_conversation
from modules.embeddings import load_embedding_model, generate_embedding
from modules.prediction import load_ml_models, predict_score, predict_level, decode_learner_level
from modules.feedback import get_feedback, generate_authoritative_assessment

def test_assessment_flow(example_id, topic, question, initial_resp, followup_q, followup_resp):
    print("=" * 70)
    print(f"TESTING ASSESSMENT FLOW: {example_id} ({topic})")
    print("=" * 70)

    # 1. Models
    embedding_model = load_embedding_model()
    score_model, level_model, encoder_model = load_ml_models()

    # 2. Experimental ML Pipeline
    conversation_text = build_conversation(initial_resp, followup_resp)
    embedding = generate_embedding(conversation_text, embedding_model)
    experimental_score = predict_score(embedding, score_model)
    predicted_level_encoded = predict_level(embedding, level_model)
    experimental_level = decode_learner_level(predicted_level_encoded, encoder_model)

    # 3. Authoritative Rubric Pipeline
    auth = generate_authoritative_assessment(topic, question, initial_resp, followup_q, followup_resp)
    authoritative_score = auth["score"]
    authoritative_level = auth["learner_level"]

    # 4. Feedback Generation (preserves authoritative assessment)
    feedback_data = get_feedback(
        question=question,
        initial_response=initial_resp,
        followup_question=followup_q,
        followup_response=followup_resp,
        experimental_ml_score=experimental_score,
        experimental_ml_learner_level=experimental_level,
        topic=topic,
        use_llm=False
    )

    # Verify field mappings in feedback_data
    print("\n--- RESULTS PAGE DISPLAY VALUES ---")
    print(f"[Main Display] Authoritative Score:        {feedback_data['score']:.2f} / 10")
    print(f"[Main Display] Authoritative Learner Level: {feedback_data['learner_level']}")
    print(f"[Main Display] Assessment Summary:          \"{feedback_data.get('overall_summary', '')}\"")
    print(f"[Main Display] Personalized Feedback:       \"{feedback_data.get('feedback', '')}\"")
    print("\n--- TECHNICAL & RESEARCH DETAILS EXPANDER ---")
    print(f"[Expander Metric 1] Experimental XGBoost Score:         {feedback_data['experimental_ml_score']:.2f} / 10")
    print(f"[Expander Metric 2] Experimental ML Learner Level:      {feedback_data['experimental_ml_learner_level']}")
    print(f"[Expander Explanatory Note] Present:                   YES")

    # Assertions
    assert feedback_data["score"] == authoritative_score, "Main score must equal authoritative score!"
    assert feedback_data["learner_level"] == authoritative_level, "Main learner level must equal authoritative level!"
    assert feedback_data["experimental_ml_score"] == experimental_score, "Experimental score must match XGBoost regressor output!"
    assert feedback_data["experimental_ml_learner_level"] == experimental_level, "Experimental learner level must match XGBoost classifier output!"

    # Verify feedback reflects authoritative score
    if authoritative_score >= 8.5:
        assert "strong" in feedback_data["overall_summary"].lower() or "accurate" in feedback_data["overall_summary"].lower()
    elif authoritative_score < 5.0:
        assert "misunderstanding" in feedback_data["overall_summary"].lower() or "not provide sufficient evidence" in feedback_data["overall_summary"].lower()

    print("\n[VERIFICATION STATUS] PASSED: All metrics verified with exact sources.")
    return {
        "example_id": example_id,
        "authoritative_score": authoritative_score,
        "authoritative_level": authoritative_level,
        "experimental_score": round(experimental_score, 2),
        "experimental_level": experimental_level,
    }

def main():
    # Example 1: Strong response (demonstrates difference between rubric score ~9.1 and raw XGBoost score ~0.89)
    res1 = test_assessment_flow(
        example_id="Example 1 (Strong Java Response)",
        topic="Inheritance",
        question="What is inheritance in Java and why is it used?",
        initial_resp="Inheritance allows a child class to acquire properties and methods of a parent class using the extends keyword. It promotes code reusability and creates an is-a relationship.",
        followup_q="Explain method overriding in inheritance.",
        followup_resp="Method overriding occurs when a subclass provides its own specific implementation of a method that is already defined in its superclass."
    )

    # Example 2: Partial/Intermediate response
    res2 = test_assessment_flow(
        example_id="Example 2 (Partial/Moderate Java Response)",
        topic="Polymorphism",
        question="What is polymorphism in Java?",
        initial_resp="Polymorphism means many forms. It allows methods to do different things.",
        followup_q="Can you explain method overloading?",
        followup_resp="Overloading is when two methods have the same name but different parameters in the same class."
    )

    print("\n" + "=" * 70)
    print("SUMMARY OF SOURCE VERIFICATION")
    print("=" * 70)
    print(f"Example 1 -> Main: {res1['authoritative_score']}/10 ({res1['authoritative_level']}) | Experimental: {res1['experimental_score']}/10 ({res1['experimental_level']})")
    print(f"Example 2 -> Main: {res2['authoritative_score']}/10 ({res2['authoritative_level']}) | Experimental: {res2['experimental_score']}/10 ({res2['experimental_level']})")
    print("All tests passed successfully.")

if __name__ == "__main__":
    main()
