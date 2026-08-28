"""
Assessment module for JCAB.
Handles the controlled question bank and follow-up generation logic.
"""
from typing import Dict, Tuple

# Controlled Question Bank
QUESTION_BANK: Dict[str, Tuple[str, str]] = {
    "OOP Fundamentals": (
        "What are the core concepts of Object-Oriented Programming (OOP) in Java?",
        "How do these concepts help in writing maintainable code?"
    ),
    "Inheritance": (
        "What is inheritance in Java? Explain its purpose and how it is used.",
        "Can you explain the difference between single inheritance and multilevel inheritance?"
    ),
    "Polymorphism": (
        "What is polymorphism in Java?",
        "Can you explain runtime polymorphism with a Java example?"
    ),
    "Encapsulation": (
        "How do you achieve encapsulation in Java and why is it important?",
        "What is the role of access modifiers in encapsulation?"
    ),
    "Exception Handling": (
        "What is the difference between checked and unchecked exceptions in Java?",
        "Can you provide an example of when you would use a custom exception?"
    ),
    "Multithreading": (
        "How do you create a thread in Java?",
        "What is the difference between extending the Thread class and implementing the Runnable interface?"
    ),
    "Collections": (
        "Explain the difference between List, Set, and Map in the Java Collections Framework.",
        "When would you choose to use a HashMap over a TreeMap?"
    ),
    "JVM Architecture": (
        "What is the role of the JVM in Java's platform independence?",
        "Can you briefly explain the different memory areas within the JVM?"
    ),
    "Garbage Collection": (
        "How does Garbage Collection work in Java?",
        "What does the 'finalize()' method do, and why is it generally discouraged in modern Java?"
    ),
    "Interfaces & Abstract Classes": (
        "What are the key differences between an interface and an abstract class in Java?",
        "When designing a system, how do you decide whether to use an interface or an abstract class?"
    )
}

def get_assessment_questions(topic: str) -> Tuple[str, str]:
    """
    Retrieves the initial question and follow-up question for a given topic.
    If the topic is not in the bank, returns generic fallback questions.
    """
    return QUESTION_BANK.get(
        topic, 
        (
            f"Please explain the core concepts of {topic} in Java.",
            "Can you provide a specific example of how that is used in practice?"
        )
    )
