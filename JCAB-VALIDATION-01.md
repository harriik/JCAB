# JCAB-VALIDATION-01: Real MCA Student Validation Setup

## 1. Objective
The objective of this phase is to evaluate the JCAB framework's personalized feedback generation using real-world data from MCA (Master of Computer Applications) students, avoiding the limitations of perfectly separable synthetic data.

## 2. Participant Criteria
- **Target:** 10–20 MCA students (Pilot Phase).
- **Domain:** Java Programming.
- **Privacy:** Anonymous participation is supported and encouraged. No mandatory PII is collected.

## 3. Procedure
1. **Consent:** Students agree to voluntary research participation.
2. **Assessment:** Students engage in a 2-stage conversational assessment (Initial + Follow-up) per concept.
3. **Feedback:** Students receive XGBoost-scored, Multi-Agent LLM-generated personalized feedback.
4. **Evaluation:** Students complete a 5-question Likert scale survey (1-5) rating relevance, helpfulness, specificity, recommendation utility, and usability, alongside two open text questions.

## 4. Data Collected
- **Assessment Logic:** (Question, Initial Response, Follow-up, Final Response)
- **ML Outputs:** (Predicted Score, Learner Level, Generated Feedback)
- **Survey Ratings:** (Relevance, Helpfulness, Specificity, Utility, Usability)
- **Qualitative:** (What was useful, what needs improvement)

*Note: API keys are securely kept out of all logs and exports.*

## 5. Limitations & Distinctions
**IMPORTANT RESEARCH DISTINCTION:**
The metrics derived from this real-student validation must **never** be combined with:
- **A. Synthetic Development Evaluation:** (Proved models function, but data was trivially separable).
- **B. Public SciEntsBank Benchmark:** (Historical standard benchmark for general science).

Real student conversational data is expected to be substantially noisier. The goal of this validation is to test the *human-perceived utility* of the LLM feedback layer, not to chase 100% classification accuracy.

## 6. Analysis Plan
Data will be collected in `research/validation/student_validation_raw.csv` and `research/validation/student_feedback_ratings.csv`. The `analyze_validation.py` script will compute aggregate ratings and identify thematic qualitative feedback to conclude the pilot study.
