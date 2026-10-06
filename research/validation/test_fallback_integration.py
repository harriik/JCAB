import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.feedback import get_feedback

def main():
    topic = "Concurrency vs Parallelism"
    question = "What is the difference between concurrency and parallelism in Java?"
    followup_q = "How does the Fork/Join framework help achieve parallelism?"
    
    init_resp = "Concurrency is about structure—making progress on many tasks at once. Parallelism is about execution—actually running tasks at the same time. In Java, concurrency is achieved with threads and executors, while parallelism leverages multi-core processors with tools like Fork/Join and parallel streams."
    
    followup_resp = "The Fork/Join framework in Java is designed specifically to help achieve parallelism by breaking large tasks into smaller subtasks and executing them across multiple processor cores. It's part of the java.util.concurrent package and was introduced in Java 7."
    
    # We pretend the XGBoost predicted score was 1.97
    result = get_feedback(
        question=question,
        initial_response=init_resp,
        followup_question=followup_q,
        followup_response=followup_resp,
        score=1.97,
        learner_level=1, # Doesn't matter
        topic=topic,
        use_llm=False
    )
    
    print("="*40)
    print("GET_FEEDBACK RESULT")
    print("="*40)
    for k, v in result.items():
        print(f"{k}: {v}")
    print("="*40)

if __name__ == "__main__":
    main()
