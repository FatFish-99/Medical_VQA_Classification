import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Load your actual dataset containing true labels and predictions
df = pd.read_csv("data/results.csv")

# Convert the continuous probabilities into binary predictions at a 0.5 threshold
df["cnn_pred"] = (df["cnn_prob"] >= 0.5).astype(int)
df["fusion_pred"] = (df["fusion_prob"] >= 0.5).astype(int)

results = []

models = {
    "CNN Only": ("cnn_pred", "cnn_prob"),
    "Fusion Model": ("fusion_pred", "fusion_prob")
}

for model_name, (pred_col, prob_col) in models.items():
    accuracy = accuracy_score(df["true_label"], df[pred_col])
    f1 = f1_score(df["true_label"], df[pred_col])
    auc = roc_auc_score(df["true_label"], df[prob_col])

    results.append({
        "Model": model_name,
        "Accuracy": round(accuracy, 4),
        "F1 Score": round(f1, 4),
        "AUC-ROC": round(auc, 4)
    })

results_df = pd.DataFrame(results)

# Save the metrics table directly to your outputs folder
results_df.to_csv("outputs/tables/model_comparison.csv", index=False)

print("\n--- Model Evaluation Results ---")
print(results_df)