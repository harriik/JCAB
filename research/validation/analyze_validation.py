import pandas as pd
from pathlib import Path
import json

import sys

def analyze_validation(dir_path="d:/JCAB/research/validation"):
    val_dir = Path(dir_path)
    raw_csv = val_dir / "student_validation_raw.csv"
    survey_csv = val_dir / "student_feedback_ratings.csv"
    
    # If using synthetic, filenames are different
    if "synthetic" in str(val_dir):
        raw_csv = val_dir / "synthetic_student_validation_raw.csv"
        survey_csv = val_dir / "synthetic_student_feedback_ratings.csv"
    
    if not raw_csv.exists() or not survey_csv.exists():
        print(f"Validation data not found in {val_dir}. Ensure pilot data is collected first.")
        return
        
    try:
        df_raw = pd.read_csv(raw_csv)
    except Exception as e:
        print(f"Error reading raw CSV: {e}")
        return
        
    try:
        df_survey = pd.read_csv(survey_csv)
    except Exception as e:
        print(f"Error reading survey CSV: {e}")
        return
    
    participants = df_survey['participant_id'].nunique()
    assessments = len(df_raw)
    
    # Exclude TEST participants if any
    test_mask = df_survey['participant_id'].str.upper().str.contains('TEST')
    df_survey_clean = df_survey[~test_mask]
    
    if len(df_survey_clean) == 0:
        print("No valid (non-TEST) survey data found.")
        return
    
    print("=== MCA Student Validation Summary ===")
    print(f"Total Unique Participants: {participants}")
    print(f"Total Assessment Interactions: {assessments}")
    
    # Exclude TEST participants from raw data as well
    test_mask_raw = df_raw['participant_id'].astype(str).str.upper().str.contains('TEST')
    df_raw_clean = df_raw[~test_mask_raw]
    
    print("\n--- Feedback Mode Distribution ---")
    if 'feedback_mode' in df_raw_clean.columns:
        modes = df_raw_clean['feedback_mode'].value_counts()
        for mode, count in modes.items():
            print(f"- {mode}: {count}")
    else:
        print("feedback_mode column not found.")
    
    rating_cols = ['q1_relevant', 'q2_helpful', 'q3_specific', 'q4_recommendation', 'q5_usability']
    avg_ratings = df_survey_clean[rating_cols].mean()
    
    print("\n--- Average Ratings (1-5) ---")
    print(f"1. Relevant to Answer: {avg_ratings['q1_relevant']:.2f}")
    print(f"2. Helpful for Weakness: {avg_ratings['q2_helpful']:.2f}")
    print(f"3. Specific to Response: {avg_ratings['q3_specific']:.2f}")
    print(f"4. Useful Recommendation: {avg_ratings['q4_recommendation']:.2f}")
    print(f"5. Easy to Use: {avg_ratings['q5_usability']:.2f}")
    
    overall_avg = avg_ratings.mean()
    print(f"\nOverall Average Rating: {overall_avg:.2f} / 5.0")
    
    print("\n--- Rating Distribution ---")
    all_ratings = df_survey_clean[rating_cols].values.flatten()
    import collections
    dist = collections.Counter(all_ratings)
    total_ratings = len(all_ratings)
    for score in range(5, 0, -1):
        count = dist.get(score, 0)
        pct = (count / total_ratings * 100) if total_ratings > 0 else 0
        print(f"{score} Stars: {count} ({pct:.1f}%)")
    
    print("\n--- Qualitative Feedback Themes ---")
    useful_comments = df_survey_clean['most_useful'].dropna().tolist()
    improve_comments = df_survey_clean['needs_improvement'].dropna().tolist()
    
    print("Most Useful Comments (Sample):")
    for c in useful_comments[:3]:
        print(f"- {c}")
        
    print("\nNeeds Improvement Comments (Sample):")
    for c in improve_comments[:3]:
        print(f"- {c}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        analyze_validation(sys.argv[1])
    else:
        analyze_validation()
