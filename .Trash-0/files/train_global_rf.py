import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings("ignore")

print("[*] INITIATING GLOBAL RANDOM FOREST & SHAP PIPELINE...")

processed_dir = "/workspace/data/processed/geo_pool"
matrix_path = os.path.join(processed_dir, "X_matrix_labeled.csv")
labels_path = os.path.join(processed_dir, "y_labels.csv")

# 1. Load the Ground Truth Data
print("[*] Loading Ground Truth Matrix and Labels...")
X_full = pd.read_csv(matrix_path, index_col=0).T 
y_full = pd.read_csv(labels_path, index_col=0)['mortality'].astype(int)

# Double check alignment
assert all(X_full.index == y_full.index), "[!] CRITICAL ERROR: Patients in Matrix and Labels do not align!"

print(f"[+] Loaded successfully: {X_full.shape[0]} Patients, {X_full.shape[1]} Genes.")
print(f"    -> Mortality Distribution: {sum(y_full==0)} Survived, {sum(y_full==1)} Died.")

# 2. Train / Test Split
X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)

# 3. Train the Random Forest Surrogate
print("\n[*] Training Random Forest Surrogate Model (Hunting for Universal Biomarkers)...")

rf_model = RandomForestClassifier(
    n_estimators=500, 
    max_depth=5,              # Shallow trees prevent memorization of 8,400 genes
    max_features='sqrt',      # Force the AI to sample different genes per split
    class_weight='balanced',  # Automatically penalize mistakes on the minority (Death) class
    random_state=42,
    n_jobs=-1                 # Max out the CPU cores
)

# Fit the model
rf_model.fit(X_train, y_train)

# 4. Evaluate Performance
val_preds = rf_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_preds)
print(f"\n[+] Surrogate Model Validation AUROC: {val_auc:.4f}")
print("\nClassification Report (Validation Set):")
print(classification_report(y_val, (val_preds > 0.5).astype(int)))

# 5. SHAP Interpretability (The Magic)
print("\n[*] Calculating Global SHAP Game-Theory Values...")
explainer = shap.TreeExplainer(rf_model)
shap_values = explainer.shap_values(X_train)

# --- THE FIX: Slicing the 3D SHAP Cube ---
# If SHAP returns a 3D matrix (Patients x Genes x Classes), we slice out Class 1 (Death)
if isinstance(shap_values, list):
    shap_2d = shap_values[1]
elif len(np.shape(shap_values)) == 3:
    shap_2d = shap_values[:, :, 1]
else:
    shap_2d = shap_values

# 6. Generate the Plot
print("[+] Generating Figure: Global Biomarker Signatures...")
plt.figure(figsize=(12, 10))
plt.title(f"GLOBAL SHAP Feature Importance (RF Surrogate)\n(Universal Mortality Biomarkers across {X_full.shape[0]} Patients)", fontsize=16, pad=20)

# Plot using the explicitly flattened 2D matrix
shap.summary_plot(shap_2d, X_train, max_display=20, show=False)

plot_path = os.path.join(processed_dir, "Real_Global_SHAP_Summary_RF.png")
plt.tight_layout()
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\n[!!!] High-Resolution Global Plot saved to: {plot_path}")