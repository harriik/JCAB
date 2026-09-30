# JCAB-EVAL-01: Final Research Evaluation

## 1. Evaluation Objective
The objective of this evaluation is to establish the preliminary technical validity of the JCAB (Java Conversational Assessment Bot) architecture prior to real-world MCA student deployment. 

## 2. Dataset Description
**IMPORTANT LIMITATION:** The 5,000-row Java dataset utilized to train the current XGBoost and Sentence-BERT components is a **synthetic development dataset**. 
- It simulates conversational assessment data (Initial Response + Follow-up Response) across 50 Java topics.
- It is designed strictly for system pipeline development and architecture validation.
- **Under no circumstances should these metrics be presented as evidence of real student performance or real-world pedagogical efficacy.** 

## 3. Provenance & Component Tracking
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Score Model:** `models/xgb_regressor.pkl`
- **Level Model:** `models/xgb_classifier.pkl`
- **Generative Feedback:** Gemini API (`gemini-3.6-flash`)
- **Timestamp:** 2026-09-24

## 4. Evaluation Breakdown

### A. SciEntsBank Benchmark
*Preliminary historical benchmarks from the SciEntsBank Test-UA split.*
- **Random Forest (Baseline):** Accuracy ≈ 0.5722, Macro F1 ≈ 0.4241
- **XGBoost (Baseline):** Accuracy ≈ 0.4907, Macro F1 ≈ 0.3484

### B. Synthetic Java Development Evaluation
**IMPORTANT AUDIT NOTE:** The original `xgb_regressor.pkl` and `xgb_classifier.pkl` were evaluated directly on the training set resulting in train-test leakage (In-Sample). To ensure validity, two rigorous hold-out strategies were subsequently executed on the dataset.

#### B.1. Row-Level Hold-Out (80/20 Stratified)
*Evaluating freshly trained models on an 80% row-level split (4,000 rows) and tested on a held-out 20% split (1,000 rows).*
- **XGBoost Held-Out Score Predictor:** MAE = 0.0001, RMSE = 0.0002
- **XGBoost Held-Out Level Classifier:** Accuracy = 1.0000, Macro Precision = 1.0000, Macro Recall = 1.0000, Macro F1 = 1.0000

#### B.2. Unseen-Question Hold-Out (Grouped Split)
*Testing whether the model generalizes to completely unseen questions by grouping on `Question_ID`. 80% of unique questions (40 questions, 4000 rows) were used for training, and 20% (10 questions, 1000 rows) were strictly held out for testing.*
- **XGBoost Unseen-Question Score Predictor:** MAE = 0.0001, RMSE = 0.0001
- **XGBoost Unseen-Question Level Classifier:** Accuracy = 1.0000, Macro Precision = 1.0000, Macro Recall = 1.0000, Macro F1 = 1.0000

#### B.3. Evaluation Comparison & Synthetic Limitations
The purpose of the grouped experiment is to measure generalization to unseen Java questions. While a row-level split can place variants of the same question in both train and test sets, the Unseen-Question split strictly guarantees that no test question was seen during training. 

Interestingly, the two evaluations do not differ in their near-perfect metrics. The classifier maintained 1.0000 accuracy and the regressor maintained ~0.0001 MAE even on completely unseen questions. This indicates that generalization to unseen Java questions is quantitatively supported *within the bounds of this synthetic dataset*. However, this extreme level of accuracy across both splits strongly highlights a **fatal synthetic-data limitation**: the synthetic generation process likely injected trivial separability or deterministic artifacts (such as strict keyword correlations, predictable text lengths, or explicit level markers) that the model easily memorizes. Real MCA student conversational data will be significantly noisier, and these perfect metrics should not be expected in real-world deployment.

### C. LLM Live Execution Status
*Evaluating the actual Gemini Multi-Agent output on three controlled conceptual cases.*
- **Gemini Pipeline Status:** **PENDING**
- **Reasoning:** Three controlled cases (Strong, Partial, Incorrect understanding) were executed against the Multi-Agent architecture using the correct Google Gemini API key. However, the Gemini API experienced a sustained HTTP 503 UNAVAILABLE service outage due to high demand. No live Gemini LLM outputs were successfully retrieved during this run.

### D. Fallback / Error-Handling Evaluation
*Evaluating the system's resilience to API failure.*
- **Status:** **PASS**
- **Reasoning:** Because the Gemini API was unavailable (503), the system successfully intercepted the fatal exception and executed the `Development Fallback (Rule-Based)` mechanism for all three test cases. The output recorded in `feedback_case_comparison.json` is explicitly the rule-based fallback output, demonstrating perfect failover functionality without crashing the assessment loop.

### E. Real-Student Validation
*Evaluating real MCA student performance.*
- **Status:** **PENDING**
- **Reasoning:** Real MCA student validation remains a separate required stage. Existing metrics/pipelines are strictly based on synthetic development data.

## 5. Functional Testing
14 end-to-end functional tests (TC01-TC14) were executed covering Domain Selection, Sentence-BERT processing, ML Integration, and Missing API Key failover.
- **Total Tests Passed:** 14/14 (100%)

## 6. Interpretation
The JCAB architecture is functionally complete and its error-handling mechanisms are extremely robust, successfully defending against a real-world third-party API outage. However, quantitative evaluation of the ML models and generative assessment of the LLM pipeline remain pending the availability of the raw datasets and the restoration of the Gemini API.
