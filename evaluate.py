import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os

# 1. Define paths
CSV_PATH = "data/results.csv"
OUTPUT_TABLE_PATH = "outputs/tables/model_comparison.csv"

print(f"Evaluating models from: {CSV_PATH}")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Could not find {CSV_PATH}. Please make sure the file is in the data folder.")

# Load data
df = pd.read_csv(CSV_PATH)
y_true = df["true_label"].values

# Define models to test
models = {
    "ResNet Baseline (Vision-only)": df["cnn_prob"].values,
    "Multimodal Fusion Model": df["fusion_prob"].values
}

results = []

# 2. Calculate metrics for each model
for model_name, y_prob in models.items():
    # Convert probabilities to a 0 or 1 prediction at a 0.5 threshold
    y_pred = (y_prob >= 0.5).astype(int)
    
    # Calculate metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    
    results.append({
        "Model Architecture": model_name,
        "Accuracy": f"{acc * 100:.1f}%",
        "F1-Score": f"{f1:.2f}",
        "ROC-AUC": f"{auc:.2f}"
    })

# 3. Create DataFrame and save it
df_results = pd.DataFrame(results)
os.makedirs(os.path.dirname(OUTPUT_TABLE_PATH), exist_ok=True)
df_results.to_csv(OUTPUT_TABLE_PATH, index=False)

print("\n=== Evaluation Metrics Results ===")
print(df_results.to_string(index=False))
print(f"\nSaved metrics table to: {OUTPUT_TABLE_PATH}")