import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import collections

def analyze_synthetic():
    print("==========================================================")
    print("SYNTHETIC/SIMULATED DATA — NOT REAL STUDENT VALIDATION")
    print("==========================================================\n")
    
    syn_dir = Path("d:/JCAB/research/validation/synthetic")
    raw_csv = syn_dir / "synthetic_student_validation_raw.csv"
    survey_csv = syn_dir / "synthetic_student_feedback_ratings.csv"
    plot_dir = syn_dir / "plots"
    
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    if not raw_csv.exists() or not survey_csv.exists():
        print("Data files missing. Run generate_synthetic_validation.py first.")
        return
        
    df_raw = pd.read_csv(raw_csv)
    df_survey = pd.read_csv(survey_csv)
    
    participants = df_survey['participant_id'].nunique()
    assessments = len(df_raw)
    
    print(f"Number of Simulated Participants: {participants}")
    print(f"Number of Assessment Interactions: {assessments}\n")
    
    print("--- Learner-Level Distribution ---")
    levels = df_raw['predicted_learner_level'].value_counts()
    for level, count in levels.items():
        print(f"- {level}: {count} ({count/assessments*100:.1f}%)")
        
    print("\n--- Predicted-Score Distribution ---")
    print(df_raw['predicted_score'].describe())
    
    print("\n--- Feedback-Mode Distribution ---")
    modes = df_raw['feedback_mode'].value_counts()
    for mode, count in modes.items():
        print(f"- {mode}: {count}")
        
    rating_cols = ['q1_relevant', 'q2_helpful', 'q3_specific', 'q4_recommendation', 'q5_usability']
    avg_ratings = df_survey[rating_cols].mean()
    
    print("\n--- Average Ratings (1-5) ---")
    print(f"1. Relevant to Answer: {avg_ratings['q1_relevant']:.2f}")
    print(f"2. Helpful for Weakness: {avg_ratings['q2_helpful']:.2f}")
    print(f"3. Specific to Response: {avg_ratings['q3_specific']:.2f}")
    print(f"4. Useful Recommendation: {avg_ratings['q4_recommendation']:.2f}")
    print(f"5. Easy to Use: {avg_ratings['q5_usability']:.2f}")
    
    print("\n--- Overall Rating Distribution ---")
    all_ratings = df_survey[rating_cols].values.flatten()
    dist = collections.Counter(all_ratings)
    total_ratings = len(all_ratings)
    for score in range(5, 0, -1):
        count = dist.get(score, 0)
        print(f"{score} Stars: {count} ({(count/total_ratings*100) if total_ratings > 0 else 0:.1f}%)")
        
    print("\n--- Qualitative Feedback Themes ---")
    useful_comments = df_survey['most_useful'].dropna().tolist()
    improve_comments = df_survey['needs_improvement'].dropna().tolist()
    
    print("Most Useful Comments (Sample):")
    for c in list(set(useful_comments))[:3]:
        print(f"- {c}")
        
    print("\nNeeds Improvement Comments (Sample):")
    for c in list(set(improve_comments))[:3]:
        print(f"- {c}")
        
    print("\n--- Participant-Level Summary ---")
    # interactions per participant
    p_interactions = df_raw.groupby('participant_id').size()
    print(f"Average interactions per participant: {p_interactions.mean():.1f}")
    
    # Generate Plots
    plt.figure(figsize=(10, 6))
    sns.histplot(df_raw['predicted_score'], bins=10, kde=True)
    plt.title("Synthetic Simulation: Predicted Score Distribution")
    plt.savefig(plot_dir / "score_distribution.png")
    plt.close()
    
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df_raw, x='predicted_learner_level', order=['Excellent', 'Good', 'Partial', 'Incorrect', 'Weak'])
    plt.title("Synthetic Simulation: Learner Level Distribution")
    plt.savefig(plot_dir / "learner_level_distribution.png")
    plt.close()
    
    plt.figure(figsize=(8, 6))
    avg_ratings.plot(kind='bar')
    plt.title("Synthetic Simulation: Average Rating by Criterion")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_dir / "average_rating_criterion.png")
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.hist(all_ratings, bins=[1,2,3,4,5,6], align='left', rwidth=0.8)
    plt.title("Synthetic Simulation: Overall Rating Distribution")
    plt.xticks([1,2,3,4,5])
    plt.savefig(plot_dir / "rating_distribution.png")
    plt.close()

if __name__ == "__main__":
    analyze_synthetic()
