import sys
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor, XGBClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('d:/JCAB')

print("Loading dataset...")
file_path = r"D:\JCAB\data\JCAB_Java_Conversational_Dataset_v1_1_5000_Samples_Original_13_Columns.xlsx"
df = pd.read_excel(file_path, sheet_name="All_Data")

# Ensure no NaNs
df['Initial_Response'] = df['Initial_Response'].fillna('')
df['Followup_Response'] = df['Followup_Response'].fillna('')

embeddings_cache_path = r'D:\JCAB\data\embeddings_cache.npy'
if os.path.exists(embeddings_cache_path):
    print("Loading Embeddings from cache...")
    X = np.load(embeddings_cache_path)
else:
    print("Embeddings cache not found, generating embeddings...")
    from sentence_transformers import SentenceTransformer
    conversations = df['Initial_Response'] + " [SEP] " + df['Followup_Response']
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    X = embedding_model.encode(conversations.tolist(), show_progress_bar=True)
    np.save(embeddings_cache_path, X)

y_score = df['Teacher_Score'].values
y_level = df['Learner_Level'].values

print("Encoding target variables...")
le = LabelEncoder()
y_level_encoded = le.fit_transform(y_level)

print("Performing Unseen Question (Grouped) Split...")
unique_questions = df['Question_ID'].unique()
if hasattr(unique_questions, 'to_numpy'):
    unique_questions = unique_questions.to_numpy()
unique_questions = np.array(unique_questions)
train_q, test_q = train_test_split(unique_questions, test_size=0.2, random_state=42)

train_mask = df['Question_ID'].isin(train_q)
test_mask = df['Question_ID'].isin(test_q)

X_train, X_test = X[train_mask], X[test_mask]
y_train_score, y_test_score = y_score[train_mask], y_score[test_mask]
y_train_level, y_test_level = y_level_encoded[train_mask], y_level_encoded[test_mask]

print(f"Train Questions: {len(train_q)} | Test Questions: {len(test_q)}")
print(f"Train Rows: {len(X_train)} | Test Rows: {len(X_test)}")

print("Training fresh XGBRegressor...")
regressor = XGBRegressor(random_state=42)
regressor.fit(X_train, y_train_score)

print("Training fresh XGBClassifier...")
classifier = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
classifier.fit(X_train, y_train_level)

print("Evaluating on unseen questions...")
y_pred_score = regressor.predict(X_test)
y_pred_level_encoded = classifier.predict(X_test)

y_pred_level = le.inverse_transform(y_pred_level_encoded)
y_true_level_decoded = le.inverse_transform(y_test_level)

mae = float(mean_absolute_error(y_test_score, y_pred_score))
rmse = float(np.sqrt(mean_squared_error(y_test_score, y_pred_score)))

acc = float(accuracy_score(y_true_level_decoded, y_pred_level))
prec = float(precision_score(y_true_level_decoded, y_pred_level, average='macro'))
rec = float(recall_score(y_true_level_decoded, y_pred_level, average='macro'))
f1 = float(f1_score(y_true_level_decoded, y_pred_level, average='macro'))

print(f"Unseen Question Regression -> MAE: {mae:.4f} | RMSE: {rmse:.4f}")
print(f"Unseen Question Classification -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

# Confusion Matrix
labels = le.classes_
cm = confusion_matrix(y_true_level_decoded, y_pred_level, labels=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix - Unseen Question Evaluation')
plt.ylabel('True Level')
plt.xlabel('Predicted Level')
plt.tight_layout()
plt.savefig(r'd:\JCAB\research\evaluation\unseen_question_confusion_matrix.png')
print("Saved unseen question confusion matrix.")

results = {
    "model": "XGBoost",
    "evaluation_type": "Synthetic Java Development - Unseen Question Evaluation",
    "metrics": {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "accuracy": round(acc, 4),
        "macro_precision": round(prec, 4),
        "macro_recall": round(rec, 4),
        "macro_f1": round(f1, 4)
    },
    "provenance": {
        "dataset_file": "JCAB_Java_Conversational_Dataset_v1_1_5000_Samples_Original_13_Columns.xlsx",
        "train_questions": train_q.tolist(),
        "test_questions": test_q.tolist(),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "train_test_split": "Grouped by Question_ID (80/20)",
        "models_used": "Freshly trained in-memory",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "random_seed": 42,
        "model_parameters": "XGBRegressor(random_state=42), XGBClassifier(random_state=42, eval_metric='mlogloss')",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
}

with open(r'd:\JCAB\research\evaluation\unseen_question_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

with open(r'd:\JCAB\research\evaluation\unseen_question_metadata.json', 'w', encoding='utf-8') as f:
    json.dump(results["provenance"], f, indent=2)

csv_str = f"model,evaluation_type,accuracy,macro_precision,macro_recall,macro_f1,mae,rmse\n"
csv_str += f"XGBoost,Synthetic Java Unseen Question Evaluation,{acc:.4f},{prec:.4f},{rec:.4f},{f1:.4f},{mae:.4f},{rmse:.4f}\n"

with open(r'd:\JCAB\research\evaluation\unseen_question_results.csv', 'w', encoding='utf-8') as f:
    f.write(csv_str)
    
print("Saved unseen question metrics.")
