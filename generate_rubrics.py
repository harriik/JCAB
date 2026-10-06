import json
from modules.assessment import QUESTION_BANK

rubrics = {}
for idx, (topic, questions) in enumerate(QUESTION_BANK.items()):
    q1, q2 = questions
    # Create some sensible defaults for demonstration
    expected_concepts = [topic.lower(), "java", "object oriented", "programming"]
    keywords = [topic.lower(), "class", "method", "object"]
    ref_explanation = f"This concept relates to {topic}. It is a fundamental part of Java programming."
    
    # Add specific detailed rubrics for known test cases
    if topic == "Inheritance":
        expected_concepts = ["child class", "parent class", "reusability", "extends", "method overriding"]
        keywords = ["acquire", "fields", "methods", "parent", "child", "extend", "override"]
        ref_explanation = "Inheritance allows a child class to acquire properties and behaviors from a parent class, promoting code reusability. Method overriding allows the child to provide a specific implementation."
    elif topic == "OOP Fundamentals":
        expected_concepts = ["encapsulation", "inheritance", "polymorphism", "abstraction"]
        keywords = ["object", "class", "hide", "extend", "many forms", "blueprint"]
        ref_explanation = "OOP is based on four main pillars: Encapsulation (hiding data), Inheritance (code reuse), Polymorphism (many forms), and Abstraction (hiding complexity)."
    elif topic == "Unit Testing":
        expected_concepts = ["isolate", "methods", "classes", "reliability", "regression", "junit"]
        keywords = ["test", "isolated", "unit", "junit", "regression", "reliable"]
        ref_explanation = "Unit testing focuses on isolated methods/classes to ensure they work correctly. It helps prevent regression bugs and improves reliability."
    
    rubrics[topic] = {
        "question_id": f"Q_{idx+1:03d}",
        "topic": topic,
        "expected_concepts": expected_concepts,
        "important_keywords": keywords,
        "reference_explanation": ref_explanation
    }

with open("data/java_fallback_rubrics.json", "w", encoding="utf-8") as f:
    json.dump(rubrics, f, indent=4)

print("Created data/java_fallback_rubrics.json")
