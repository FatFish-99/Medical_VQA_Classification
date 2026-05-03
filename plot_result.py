import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import os

# 1. Load the model prediction results
CSV_PATH = "data/results.csv"
FIGURES_PATH = "outputs/figures/evaluation_plots.png"

print(f"Loading predictions from: {CSV_PATH}")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"Could not find {CSV_PATH}. Please make sure the file is in the data folder.")

# Read the file
df = pd.read_csv(CSV_PATH)

# Extract correct columns from the CSV
true_labels = df["true_label"]
cnn_probs = df["cnn_prob"]
fusion_probs = df["fusion_prob"]

# 2. Calculate ROC curves and AUC scores
fpr_cnn, tpr_cnn, _ = roc_curve(true_labels, cnn_probs)
auc_cnn = auc(fpr_cnn, tpr_cnn)

fpr_fusion, tpr_fusion, _ = roc_curve(true_labels, fusion_probs)
auc_fusion = auc(fpr_fusion, tpr_fusion)

# 3. Create the plots
plt.figure(figsize=(8, 6))

# Plot Fusion Model ROC
plt.plot(fpr_fusion, tpr_fusion, color="#818cf8", lw=3, label=f"Multimodal Fusion (AUC = {auc_fusion:.2f})")

# Plot Baseline ROC
plt.plot(fpr_cnn, tpr_cnn, color="#38bdf8", lw=2, linestyle="--", label=f"ResNet Baseline (AUC = {auc_cnn:.2f})")

# Plot random guess line
plt.plot([0, 1], [0, 1], color="#64748b", lw=1, linestyle=":")

# Add plot details
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("Receiver Operating Characteristic (ROC) Curve", fontsize=14, fontweight="bold")
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, linestyle="--", alpha=0.5)

# 4. Save the plot
os.makedirs(os.path.dirname(FIGURES_PATH), exist_ok=True)
plt.savefig(FIGURES_PATH, dpi=300, bbox_inches="tight")
plt.close()

print(f"Successfully generated ROC curve plot at: {FIGURES_PATH}")