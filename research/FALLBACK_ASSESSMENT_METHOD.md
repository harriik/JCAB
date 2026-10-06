# Local Heuristic Assessment Method (Fallback)

When the Multi-Agent LLM (Gemini API) is unavailable, the system uses a deterministic, rule-based heuristic to evaluate the student's response. This ensures the application remains functional and provides logically consistent feedback.

## Scoring Formula

The score is bounded between `0.0` and `10.0`, evaluated deterministically using explicit heuristic gates combining Semantic Similarity and Concept Coverage:

### 1. Coverage Calculation
`coverage` is the ratio of key concepts found in the student's text versus the total expected concepts defined in `data/java_fallback_rubrics.json`. Prefix-stemming (first 4 characters) is used for robust matching of variants (e.g., "reuse" matches "reusability").

### 2. Semantic Similarity
`sim` is the cosine similarity score produced by the `all-MiniLM-L6-v2` model between the reference text and the student's text.

### 3. Heuristic Gates
The system filters the response through strict quality gates:
1. **Empty / Short Gate**: If the response is extremely short (<15 chars) or contains phrases like "I don't know", it immediately scores `0.0`.
2. **Irrelevance Gate**: If `coverage == 0.0` and `sim < 0.50`, the response is deemed completely off-topic (Score bounds: `0.0 - 1.0`).
3. **Incorrect Gate**: If `coverage == 0.0` and `sim < 0.65`, the response touches on similar language but misses all core concepts, indicating confusion (Score bounds: `2.0 - 2.3`).
4. **Partial Gate**: If `coverage > 0.0` but remains low, or `sim` remains low, a baseline score is assigned (Score bounds: `3.5 - 7.0`).
5. **Incomplete Correct Gate**: If coverage is good but not exhaustive (`coverage < 0.8`), medium-high points are assigned (Score bounds: `7.0 - 9.0`).
6. **Strong Correct Gate**: If coverage is exhaustive (`coverage >= 0.8`), high points are assigned (Score bounds: `9.0 - 10.0`).

### 4. Follow-up Bonus
If the follow-up response exceeds 20 characters and contains targeted keyword stems, `+1.0` is added to the score. If it has high semantic similarity (`sim >= 0.60`) without exact keywords, `+0.5` is added.

## Learner Level Thresholds
The learner level is deterministically derived from the final heuristic score to guarantee consistency:
- `8.5 - 10.0` → **Excellent**
- `7.0 - 8.49` → **Good**
- `5.0 - 6.99` → **Partial**
- `2.5 - 4.99` → **Weak**
- `0.0 - 2.49` → **Incorrect**

## Feedback Generation
- **Strengths:** Generated based on which `expected_concepts` were detected.
- **Areas to Improve:** Generated based on which `expected_concepts` were missed.
- **Feedback:** A synthesized string highlighting the score range and conceptual coverage.
- **Recommendation:** Tailored based on the learner level (e.g., weak -> review fundamentals; excellent -> advance to complex topics).
