import os
import sys
import json
sys.path.append('d:/JCAB')

# API key should be set in environment variables or .env file

from modules.feedback import get_feedback
from config.settings import USE_LLM_FEEDBACK

cases = [
    {
        "id": "CASE_1",
        "type": "Strong conceptual understanding",
        "question": "What is polymorphism in Java?",
        "initial_response": "Polymorphism means many forms. It allows objects to be treated as instances of their parent class.",
        "followup_question": "Can you explain method overriding vs overloading?",
        "followup_response": "Overloading is compile-time polymorphism where methods have same name but different parameters. Overriding is runtime where a child class provides a specific implementation of a parent's method.",
        "score": 9.2,
        "learner_level": 2, # Advanced
        "topic": "Polymorphism"
    },
    {
        "id": "CASE_2",
        "type": "Partial conceptual understanding",
        "question": "What is polymorphism in Java?",
        "initial_response": "It means methods can do different things.",
        "followup_question": "Can you explain method overriding vs overloading?",
        "followup_response": "Overloading is when you have different parameters, but I am not sure about overriding.",
        "score": 5.5,
        "learner_level": 1, # Intermediate
        "topic": "Polymorphism"
    },
    {
        "id": "CASE_3",
        "type": "Incorrect / misconception",
        "question": "What is polymorphism in Java?",
        "initial_response": "It's when variables change their data types automatically.",
        "followup_question": "Can you explain method overriding vs overloading?",
        "followup_response": "Compile time is when the compiler catches syntax errors.",
        "score": 1.5,
        "learner_level": 0, # Beginner
        "topic": "Polymorphism"
    }
]

results = []

for case in cases:
    print(f"Running {case['id']}...")
    res = get_feedback(
        question=case["question"],
        initial_response=case["initial_response"],
        followup_question=case["followup_question"],
        followup_response=case["followup_response"],
        score=case["score"],
        learner_level=case["learner_level"],
        topic=case["topic"],
        use_llm=True
    )
    results.append({
        "case": case["id"],
        "type": case["type"],
        "inputs": case,
        "final_output": res
    })

with open('d:/JCAB/research/evaluation/feedback_case_comparison.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

print("Done!")
