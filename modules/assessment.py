"""
Assessment module for JCAB.
Handles the controlled question bank and follow-up generation logic.
"""
from typing import Dict, Tuple

import random
from typing import Dict, Tuple, List

# Expanded Controlled Question Bank (50 Topics/Questions for Java Conceptual Assessment)
QUESTION_BANK: Dict[str, Tuple[str, str]] = {
    "OOP Fundamentals": ("What are the core concepts of Object-Oriented Programming (OOP) in Java?", "How do these concepts help in writing maintainable code?"),
    "Inheritance": ("What is inheritance in Java? Explain its purpose and how it is used.", "Can you explain the difference between single inheritance and multilevel inheritance?"),
    "Polymorphism": ("What is polymorphism in Java?", "Can you explain runtime polymorphism with a Java example?"),
    "Encapsulation": ("How do you achieve encapsulation in Java and why is it important?", "What is the role of access modifiers in encapsulation?"),
    "Exception Handling": ("What is the difference between checked and unchecked exceptions in Java?", "Can you provide an example of when you would use a custom exception?"),
    "Multithreading": ("How do you create a thread in Java?", "What is the difference between extending the Thread class and implementing the Runnable interface?"),
    "Collections Framework": ("Explain the difference between List, Set, and Map in the Java Collections Framework.", "When would you choose to use a HashMap over a TreeMap?"),
    "JVM Architecture": ("What is the role of the JVM in Java's platform independence?", "Can you briefly explain the different memory areas within the JVM?"),
    "Garbage Collection": ("How does Garbage Collection work in Java?", "What does the 'finalize()' method do, and why is it generally discouraged in modern Java?"),
    "Interfaces vs Abstract Classes": ("What are the key differences between an interface and an abstract class in Java?", "When designing a system, how do you decide whether to use an interface or an abstract class?"),
    "String Pool": ("What is the String Pool in Java?", "Why are strings immutable in Java, and how does the string pool leverage this?"),
    "Generics": ("What are Generics in Java and what problem do they solve?", "Can you explain type erasure in the context of Java Generics?"),
    "Lambda Expressions": ("What are Lambda expressions in Java 8?", "How do Lambda expressions relate to Functional Interfaces?"),
    "Stream API": ("What is the purpose of the Stream API in Java 8?", "Can you explain the difference between intermediate and terminal operations in Streams?"),
    "Optional Class": ("What is the Optional class in Java and why was it introduced?", "How does using Optional prevent NullPointerExceptions compared to traditional null checks?"),
    "Concurrency vs Parallelism": ("What is the difference between concurrency and parallelism in Java?", "How does the Fork/Join framework help achieve parallelism?"),
    "Synchronization": ("What does the 'synchronized' keyword do in Java?", "Explain the concept of a thread deadlock and how it can be prevented."),
    "Volatile Keyword": ("What is the 'volatile' keyword used for in Java?", "How does 'volatile' differ from 'synchronized' in terms of thread safety?"),
    "Static Keyword": ("What does the 'static' keyword mean when applied to variables and methods?", "Can a static method access non-static variables? Explain why or why not."),
    "Final Keyword": ("What is the purpose of the 'final' keyword in Java?", "Explain how 'final' behaves differently when applied to a variable, a method, and a class."),
    "Constructors": ("What is a constructor in Java?", "Can you explain constructor overloading and how the 'this()' keyword is used?"),
    "Method Overloading vs Overriding": ("What is the difference between method overloading and method overriding?", "Can you override a private method in Java? Why or why not?"),
    "Super Keyword": ("What is the 'super' keyword used for in Java?", "How do you use 'super' to invoke a parent class constructor?"),
    "Wrapper Classes": ("What are wrapper classes in Java and why are they necessary?", "Explain the concepts of autoboxing and unboxing."),
    "Serialization": ("What is object serialization in Java?", "What is the purpose of the 'transient' keyword during serialization?"),
    "Cloning": ("What is object cloning in Java?", "Explain the difference between a shallow copy and a deep copy."),
    "ClassLoaders": ("What is a ClassLoader in Java?", "Can you name the different types of built-in ClassLoaders in the JVM?"),
    "Reflection API": ("What is the Java Reflection API used for?", "What are the security and performance implications of using Reflection?"),
    "Annotations": ("What are annotations in Java?", "Can you explain the difference between built-in annotations and custom annotations?"),
    "Enums": ("What is an Enum in Java?", "How can you add methods and constructors to a Java Enum?"),
    "Memory Leaks": ("Since Java has a Garbage Collector, can memory leaks still occur?", "Provide an example of how a memory leak might happen in a Java application."),
    "Comparable vs Comparator": ("What is the difference between Comparable and Comparator interfaces?", "When would you choose to implement Comparator instead of Comparable?"),
    "Hashcode and Equals": ("What is the contract between the 'hashCode()' and 'equals()' methods?", "What happens if you override 'equals()' but forget to override 'hashCode()'?"),
    "Default Methods": ("What are default methods in Java interfaces?", "How do default methods resolve the diamond problem of multiple inheritance in Java?"),
    "Type Inference": ("What is type inference in Java?", "How does the 'var' keyword (introduced in Java 10) work with type inference?"),
    "Records": ("What are Java Records (introduced in Java 14)?", "How do Records reduce boilerplate code compared to traditional POJOs?"),
    "Sealed Classes": ("What are Sealed Classes in Java?", "How do sealed classes provide better control over inheritance hierarchies?"),
    "Pattern Matching": ("What is Pattern Matching for 'instanceof' in Java?", "How does pattern matching improve code readability and safety?"),
    "JDBC": ("What is JDBC and what is its role in Java applications?", "Explain the difference between Statement and PreparedStatement."),
    "Design Patterns (Singleton)": ("What is the Singleton design pattern?", "How do you implement a thread-safe Singleton in Java?"),
    "Design Patterns (Factory)": ("What is the Factory design pattern?", "When would you use a Factory pattern over a simple constructor?"),
    "Design Patterns (Observer)": ("What is the Observer design pattern?", "Can you provide a real-world example of where the Observer pattern is used?"),
    "SOLID Principles": ("What are the SOLID principles of object-oriented design?", "Can you explain the Single Responsibility Principle with a Java example?"),
    "Dependency Injection": ("What is Dependency Injection (DI)?", "How does Dependency Injection promote loose coupling in Java applications?"),
    "Unit Testing": ("What is unit testing and why is it important in Java development?", "How do mocking frameworks like Mockito help in unit testing?"),
    "Build Tools": ("What is the purpose of build tools like Maven or Gradle in Java?", "Explain the concept of dependency management in Maven."),
    "REST APIs": ("What is a REST API and how is it typically implemented in Java?", "What are the common HTTP methods used in REST APIs?"),
    "Spring Framework": ("What is the Spring Framework in Java?", "Explain the concept of Inversion of Control (IoC) in the context of Spring."),
    "Microservices": ("What is a microservices architecture?", "What are the benefits and challenges of building microservices in Java?"),
    "JPA / Hibernate": ("What is JPA (Java Persistence API) and how does it relate to Hibernate?", "What is the difference between a transient, persistent, and detached entity state?")
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

def get_random_questions(count: int = 1) -> List[Dict[str, str]]:
    """
    Dynamically fetches N random questions from the 50-question bank.
    Returns a list of dicts with 'topic', 'question', 'followup'.
    """
    topics = list(QUESTION_BANK.keys())
    selected_topics = random.sample(topics, min(count, len(topics)))
    
    questions = []
    for t in selected_topics:
        q, f = QUESTION_BANK[t]
        questions.append({
            "topic": t,
            "question": q,
            "followup": f
        })
    return questions
