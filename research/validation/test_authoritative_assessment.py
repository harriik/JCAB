import os
import sys
from pathlib import Path
from dotenv import load_dotenv

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
load_dotenv(".env")

from modules.feedback import generate_authoritative_assessment
from modules.embeddings import load_embedding_model, generate_embedding
from modules.prediction import load_ml_models, predict_score, predict_level, decode_learner_level

def run_test():
    print("==================================================")
    print("      AUTHORITATIVE ASSESSMENT TEST")
    print("==================================================")

    # Load experimental ML models
    embedding_model = load_embedding_model()
    score_model, level_model, encoder_model = load_ml_models()
    
    test_cases = [
        {
            "name": "1. Strong Java Records answer",
            "topic": "Java Records",
            "question": "What are Java Records (introduced in Java 14)?",
            "initial_response": "Java Records, introduced as a preview feature in Java 14 and standardized in Java 16, are a special kind of class designed to make data-carrying objects concise, immutable, and easy to use. They reduce boilerplate code when you just need a class to hold values.",
            "followup_question": "How do Records reduce boilerplate code compared to traditional POJOs?",
            "followup_response": "Java Records reduce boilerplate code compared to traditional POJOs (Plain Old Java Objects) by automatically generating the methods and structure you’d normally have to write manually."
        },
        {
            "name": "2. Strong Concurrency vs Parallelism answer",
            "topic": "Concurrency vs Parallelism",
            "question": "What is the difference between concurrency and parallelism in Java?",
            "initial_response": "Concurrency is about structure—making progress on many tasks at once. Parallelism is about execution—actually running tasks at the same time.",
            "followup_question": "How does the Fork/Join framework help achieve parallelism?",
            "followup_response": "The Fork/Join framework splits tasks into smaller subtasks, runs them in parallel using a thread pool, and joins the results."
        },
        {
            "name": "3. Correct but incomplete answer",
            "topic": "Java Interfaces",
            "question": "What is an interface in Java?",
            "initial_response": "An interface is a contract that classes can implement.",
            "followup_question": "Can interfaces have method implementations?",
            "followup_response": "Yes, starting in Java 8 they can have default methods."
        },
        {
            "name": "4. Partial answer",
            "topic": "Inheritance",
            "question": "What is inheritance?",
            "initial_response": "Inheritance is when one class extends another.",
            "followup_question": "Why is it useful?",
            "followup_response": "It lets you reuse code."
        },
        {
            "name": "5. Incorrect answer",
            "topic": "Polymorphism",
            "question": "What is polymorphism?",
            "initial_response": "Polymorphism means hiding data inside a class.",
            "followup_question": "Give an example.",
            "followup_response": "Making variables private."
        },
        {
            "name": "6. Irrelevant answer",
            "topic": "Exceptions",
            "question": "What is the difference between checked and unchecked exceptions?",
            "initial_response": "I like writing code in Python better than Java.",
            "followup_question": "What about exceptions?",
            "followup_response": "Exceptions are errors."
        },
        {
            "name": "7. Empty answer",
            "topic": "Generics",
            "question": "Why use Generics in Java?",
            "initial_response": "I don't know.",
            "followup_question": "Can you guess?",
            "followup_response": "No."
        }
    ]

    for tc in test_cases:
        print(f"\n--- {tc['name']} ---")
        print(f"Question: {tc['question']}")
        
        # Authoritative Assessment
        auth = generate_authoritative_assessment(
            topic=tc['topic'],
            question=tc['question'],
            initial_response=tc['initial_response'],
            followup_question=tc['followup_question'],
            followup_response=tc['followup_response']
        )
        
        # Experimental ML Assessment
        full_text = f"Student: {tc['initial_response']} Student: {tc['followup_response']}"
        embedding = generate_embedding(full_text, embedding_model)
        ml_score = predict_score(embedding, score_model)
        ml_level_enc = predict_level(embedding, level_model)
        ml_level = decode_learner_level(ml_level_enc, encoder_model)
        
        print(f"Authoritative Score: {auth['score']:.2f} / 10")
        print(f"Authoritative Learner Level: {auth['learner_level']}")
        print(f"Experimental XGBoost Score: {ml_score:.2f} / 10")
        print(f"Experimental XGBoost Learner Level: {ml_level}")

if __name__ == "__main__":
    run_test()
