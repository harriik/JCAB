import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

from modules.feedback import generate_development_fallback_feedback

def test_case(name, topic, question, initial, followup_q, followup_a):
    print("=" * 60)
    print(f"CASE: {name}")
    print("-" * 60)
    print(f"Topic: {topic}")
    print(f"Q: {question}")
    print(f"A: {initial}")
    print(f"FQ: {followup_q}")
    print(f"FA: {followup_a}")
    
    result = generate_development_fallback_feedback(
        topic=topic,
        question=question,
        initial_response=initial,
        followup_question=followup_q,
        followup_response=followup_a
    )
    
    print("-" * 60)
    print(f"Score: {result['score']} / 10")
    print(f"Learner Level: {result['learner_level']}")
    print(f"[DEBUG] Sim: {result['debug_sim']} | Cov: {result['debug_cov']}")
    print(f"Strengths: {result['strengths']}")
    print(f"Missing/Improvements: {result['areas_to_improve']}")
    print(f"Recommendation: {result['recommendation']}")
    print(f"Feedback/Summary: {result['overall_summary']}")
    print("=" * 60 + "\n")
    return result

def main():
    topic = "Concurrency vs Parallelism"
    question = "What is the difference between concurrency and parallelism in Java?"
    followup_q = "How does the Fork/Join framework help achieve parallelism?"
    
    # 1. Strong correct answer (the one from the user's report)
    c1 = test_case("1. Strong correct answer", topic, question, 
        "Concurrency is about structure—making progress on many tasks at once. Parallelism is about execution—actually running tasks at the same time. In Java, concurrency is achieved with threads and executors, while parallelism leverages multi-core processors with tools like Fork/Join and parallel streams.",
        followup_q,
        "The Fork/Join framework in Java is designed specifically to help achieve parallelism by breaking large tasks into smaller subtasks and executing them across multiple processor cores. It's part of the java.util.concurrent package and was introduced in Java 7."
    )
    
    # 2. Correct but incomplete answer
    c2 = test_case("2. Correct but incomplete answer", topic, question,
        "Concurrency is when multiple tasks are interleaved. Parallelism is when they run at the same time on different cores.",
        followup_q,
        "Fork/Join splits a big task into smaller subtasks."
    )
    
    # 3. Partially correct answer
    c3 = test_case("3. Partially correct answer", topic, question,
        "Concurrency uses threads. I am not sure about parallelism.",
        followup_q,
        "Fork join is a framework in Java."
    )
    
    # 4. Incorrect answer
    c4 = test_case("4. Incorrect answer", topic, question,
        "Concurrency is hiding data using private variables. Parallelism is inheriting from multiple classes.",
        followup_q,
        "Fork join is used to join strings together."
    )
    
    # 5. Irrelevant answer
    c5 = test_case("5. Irrelevant answer", topic, question,
        "I like to eat pizza and watch movies. Java is a coffee.",
        followup_q,
        "Single coffee is espresso, multilevel is a latte."
    )
    
    # 6. Empty/very short answer
    c6 = test_case("6. Empty/very short answer", topic, question,
        "idk",
        followup_q,
        "none"
    )
    
    # Extra topic test
    extra = test_case("7. Extra Strong Answer (Inheritance)", "Inheritance", "What is inheritance in Java? Explain its purpose and how it is used.",
        "Inheritance allows a child class to acquire properties and behaviors from a parent class using the extends keyword. It promotes code reusability.",
        "Can you explain the difference between single inheritance and multilevel inheritance?",
        "Single inheritance is when a class extends one parent class. Multilevel is when a class extends a child class, forming a chain. Also child class can use method overriding."
    )

    print("=" * 60)
    print("VERIFYING ASSERTIONS (Concurrency vs Parallelism)")
    print("-" * 60)
    s1, s2, s3, s4, s5, s6 = [c['score'] for c in (c1, c2, c3, c4, c5, c6)]
    
    print(f"Strong ({s1}) > Incomplete ({s2}): {s1 > s2}")
    print(f"Incomplete ({s2}) > Partial ({s3}): {s2 > s3}")
    print(f"Partial ({s3}) > Incorrect ({s4}): {s3 > s4}")
    print(f"Incorrect ({s4}) > Irrelevant ({s5}): {s4 > s5}")
    print(f"Irrelevant ({s5}) > Empty ({s6}): {s5 > s6}")
    
    assert s1 > s2, "Assertion Failed: strong > incomplete"
    assert s2 > s3, "Assertion Failed: incomplete > partial"
    assert s3 > s4, "Assertion Failed: partial > incorrect"
    assert s4 > s5, "Assertion Failed: incorrect > irrelevant"
    assert s5 > s6, "Assertion Failed: irrelevant > empty"
    
    print("ALL ASSERTIONS PASSED")
    print("=" * 60)

if __name__ == "__main__":
    main()

