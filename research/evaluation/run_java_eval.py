import sys
import pandas as pd
import numpy as np
import json
import joblib
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('d:/JCAB')

print("Loading dataset...")
file_path = r"D:\JCAB\data\JCAB_Java_Conversational_Dataset_v1_1_5000_Samples_Original_13_Columns.xlsx"
df = pd.read_excel(file_path, sheet_name="All_Data")

if len(df) != 5000:
    print(f"WARNING: Expected 5000 rows, found {len(df)}")
else:
    print("Verified exactly 5000 records.")

# Clean missing values just in case
df['Initial_Response'] = df['Initial_Response'].fillna('')
df['Followup_Response'] = df['Followup_Response'].fillna('')

print("Generating conversation strings...")
conversations = df['Initial_Response'] + " [SEP] " + df['Followup_Response']

import os
embeddings_cache_path = r'D:\JCAB\data\embeddings_cache.npy'
if os.path.exists(embeddings_cache_path):
    print("Loading Embeddings from cache...")
    embeddings = np.load(embeddings_cache_path)
else:
    print("Loading Embedding Model...")
    embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    print("Generating Embeddings (This may take a minute)...")
    embeddings = embedding_model.encode(conversations.tolist(), show_progress_bar=True)
    np.save(embeddings_cache_path, embeddings)

print("Loading ML Models...")
score_model = joblib.load(r"D:\JCAB\models\xgb_regressor.pkl")
if not hasattr(score_model, 'gpu_id'):
    score_model.gpu_id = None
level_model = joblib.load(r"D:\JCAB\models\xgb_classifier.pkl")
if not hasattr(level_model, 'gpu_id'):
    level_model.gpu_id = None
encoder_model = joblib.load(r"D:\JCAB\models\learner_level_encoder.pkl")

import xgboost as xgb
dmatrix = xgb.DMatrix(embeddings)
y_pred_score = score_model.get_booster().predict(dmatrix)
y_pred_probs = level_model.get_booster().predict(dmatrix)
if len(y_pred_probs.shape) > 1 and y_pred_probs.shape[1] > 1:
    y_pred_level_encoded = np.argmax(y_pred_probs, axis=1)
else:
    y_pred_level_encoded = np.round(y_pred_probs).astype(int)

y_true_score = df['Teacher_Score'].values
y_true_level = df['Learner_Level'].values

y_pred_level = encoder_model.inverse_transform(y_pred_level_encoded)

print("Calculating Metrics...")
mae = float(mean_absolute_error(y_true_score, y_pred_score))
rmse = float(np.sqrt(mean_squared_error(y_true_score, y_pred_score)))

acc = float(accuracy_score(y_true_level, y_pred_level))
prec = float(precision_score(y_true_level, y_pred_level, average='macro'))
rec = float(recall_score(y_true_level, y_pred_level, average='macro'))
f1 = float(f1_score(y_true_level, y_pred_level, average='macro'))

print(f"Regression -> MAE: {mae:.4f} | RMSE: {rmse:.4f}")
print(f"Classification -> Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | F1: {f1:.4f}")

# Generate Confusion Matrix
labels = encoder_model.classes_
cm = confusion_matrix(y_true_level, y_pred_level, labels=labels)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title('Confusion Matrix - Synthetic Java Development Evaluation')
plt.ylabel('True Level')
plt.xlabel('Predicted Level')
plt.tight_layout()
plt.savefig(r'd:\JCAB\research\evaluation\java_confusion_matrix.png')
print("Saved confusion matrix.")

results = {
    "model": "XGBoost",
    "evaluation_type": "Synthetic Java Development Evaluation",
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
        "sheet_name": "All_Data",
        "num_records": len(df),
        "train_test_split": "100% Evaluation (No split specified for this run)",
        "models_used": [
            "models/xgb_regressor.pkl",
            "models/xgb_classifier.pkl",
            "models/learner_level_encoder.pkl"
        ],
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
}

with open(r'd:\JCAB\research\evaluation\java_results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2)

csv_str = f"model,evaluation_type,accuracy,macro_precision,macro_recall,macro_f1,mae,rmse\n"
csv_str += f"XGBoost,Synthetic Java Development Evaluation,{acc:.4f},{prec:.4f},{rec:.4f},{f1:.4f},{mae:.4f},{rmse:.4f}\n"

with open(r'd:\JCAB\research\evaluation\java_results.csv', 'w', encoding='utf-8') as f:
    f.write(csv_str)
    
print("Saved metrics to JSON and CSV.")
