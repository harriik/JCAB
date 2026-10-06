# Authoritative Assessment Method

## Architecture Decision

XGBoost is retained as an experimental ML response-analysis component. Because the model was trained on synthetic development data and demonstrated inconsistent predictions on natural responses, it is not used as the authoritative student-facing grading output. The final student assessment is determined by the question-specific semantic/rubric assessment.

### 1. Authoritative Assessment
The student-facing score and learner level are generated using a deterministic Local Heuristic Assessment (`modules/feedback.py`). This evaluates:
- Semantic similarity of the student's entire response to a rubric-based reference explanation using Sentence-BERT.
- Conceptual coverage matching (identifying specific expected concepts and important keywords).
- A blended logic formula assigning a score from 0.0 to 10.0 and categorizing it as Excellent, Good, Partial, Weak, or Incorrect.

This score dictates the personalized LLM Feedback and what the student sees in their reports.

### 2. Experimental ML Analysis
The XGBoost models (both regression for score and classification for level) still run in the background. Their outputs are stored alongside the authoritative assessment as `experimental_ml_score` and `experimental_ml_learner_level` and displayed in an expandable "Experimental ML Analysis" UI component for research comparison only.

*Note: The semantic/rubric method is a heuristic logic framework and is not human-validated.*
