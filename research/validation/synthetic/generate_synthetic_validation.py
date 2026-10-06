import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random

# Ensure we can import from the root
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from modules.assessment import get_random_questions

def generate_synthetic_data():
    random.seed(42)
    np.random.seed(42)
    
    syn_dir = Path("d:/JCAB/research/validation/synthetic")
    syn_dir.mkdir(parents=True, exist_ok=True)
    
    # 20 participants
    participants = [f"SIM_P{str(i).zfill(2)}" for i in range(1, 21)]
    
    learner_profiles = ["Weak", "Partial", "Incorrect", "Good", "Excellent"]
    
    raw_rows = []
    rating_rows = []
    
    base_time = datetime(2026, 9, 25, 10, 0, 0)
    
    for p_id in participants:
        num_interactions = random.randint(2, 5)
        profile = random.choice(learner_profiles)
        
        # Ratings logic
        for i in range(num_interactions):
            # Select random question
            q = get_random_questions(1)[0]
            
            # Generate response based on profile
            if profile == "Weak":
                score = round(random.uniform(0.0, 3.5), 2)
                level = random.choice(["Weak", "Incorrect"])
                init_resp = f"I don't really know about {q['topic']}."
                followup_resp = "Can you explain it?"
                feedback = "It seems you need to review the basics of this concept."
                action = "Review introductory tutorials."
            elif profile in ["Partial", "Incorrect"]:
                score = round(random.uniform(2.5, 6.0), 2)
                level = random.choice(["Partial", "Incorrect", "Weak"])
                init_resp = f"I think {q['topic']} is about classes and methods."
                followup_resp = "It helps write code."
                feedback = "You have a partial understanding but missed key details."
                action = "Practice writing small examples."
            elif profile == "Good":
                score = round(random.uniform(6.0, 8.5), 2)
                level = random.choice(["Good", "Partial"])
                init_resp = f"The {q['topic']} is a concept in Java used to solve this."
                followup_resp = "It allows better structure."
                feedback = "Good answer, but you can add more edge cases."
                action = "Study edge cases and best practices."
            else: # Excellent
                score = round(random.uniform(8.0, 10.0), 2)
                level = random.choice(["Excellent", "Good"])
                init_resp = f"{q['topic']} is a fundamental object-oriented concept that encapsulates state and behavior."
                followup_resp = "Furthermore, it improves maintainability."
                feedback = "Excellent grasp of the concept."
                action = "Explore advanced design patterns."
                
            timestamp = base_time + timedelta(hours=random.randint(1, 48), minutes=random.randint(0, 59))
            
            raw_rows.append({
                "participant_id": p_id,
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "domain": "Java",
                "question_id": q.get('id', 'Q_UNKNOWN'),
                "topic": q['topic'],
                "assessment_question": q['question'],
                "initial_response": init_resp,
                "followup_question": q['followup'],
                "followup_response": followup_resp,
                "predicted_score": score,
                "learner_level": level,
                "feedback_mode": "Synthetic Multi-Agent LLM Simulation",
                "generated_feedback": feedback,
                "recommendation": action
            })
            
        # 1 rating per participant (usually at end of session)
        rating = 5 if profile == "Excellent" else random.randint(3, 5)
        
        def bound_rating(val):
            return max(1, min(5, val))
            
        rating_rows.append({
            "participant_id": p_id,
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "topic": q['topic'],
            "q1_relevant": bound_rating(rating + random.randint(-1, 1)),
            "q2_helpful": bound_rating(rating + random.randint(-1, 1)),
            "q3_specific": bound_rating(rating + random.randint(-1, 1)),
            "q4_recommendation": bound_rating(rating + random.randint(-1, 1)),
            "q5_usability": bound_rating(rating + random.randint(0, 1)),
            "most_useful": "The specific suggestions were great.",
            "needs_improvement": "Nothing, it was good."
        })
        
    df_raw = pd.DataFrame(raw_rows)
    df_ratings = pd.DataFrame(rating_rows)
    
    df_raw.to_csv(syn_dir / "synthetic_student_validation_raw.csv", index=False)
    df_ratings.to_csv(syn_dir / "synthetic_student_feedback_ratings.csv", index=False)
    
    print(f"Generated {len(participants)} participants.")
    print(f"Generated {len(df_raw)} assessment interactions.")

if __name__ == "__main__":
    generate_synthetic_data()
