import os
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import warnings
warnings.filterwarnings("ignore")

print("[*] INITIATING GLOBAL XGBOOST & SHAP PIPELINE...")

processed_dir = "/workspace/data/processed/geo_pool"
matrix_path = os.path.join(processed_dir, "X_matrix_labeled.csv")
labels_path = os.path.join(processed_dir, "y_labels.csv")

# 1. Load the Ground Truth Data
print("[*] Loading Ground Truth Matrix and Labels...")
X_full = pd.read_csv(matrix_path, index_col=0).T # Transpose so patients are rows, genes are columns
y_full = pd.read_csv(labels_path, index_col=0)['mortality'].astype(int)

# Double check alignment
assert all(X_full.index == y_full.index), "[!] CRITICAL ERROR: Patients in Matrix and Labels do not align!"

print(f"[+] Loaded successfully: {X_full.shape[0]} Patients, {X_full.shape[1]} Genes.")
print(f"    -> Mortality Distribution: {sum(y_full==0)} Survived, {sum(y_full==1)} Died.")

# 2. Train / Test Split
X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42, stratify=y_full)

# 3. Train the Global AI
print("\n[*] Training Global XGBoost Model (Hunting for Universal Biomarkers)...")
imbalance_ratio = sum(y_train==0) / sum(y_train==1)

xgb_model = xgb.XGBClassifier(
    n_estimators=500, 
    max_depth=4,              # Shallow trees prevent memorization
    learning_rate=0.01, 
    subsample=0.8,
    colsample_bytree=0.2,     # Force the AI to look at only 20% of genes per tree to find hidden signals
    scale_pos_weight=imbalance_ratio, 
    random_state=42,
    eval_metric="auc",
    early_stopping_rounds=50  # Stop training if the validation score stops improving
)

# Fit the model
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# 4. Evaluate Performance
val_preds = xgb_model.predict_proba(X_val)[:, 1]
val_auc = roc_auc_score(y_val, val_preds)
print(f"\n[+] Global Model Validation AUROC: {val_auc:.4f}")
print("\nClassification Report (Validation Set):")
print(classification_report(y_val, (val_preds > 0.5).astype(int)))

# 5. SHAP Interpretability (The Magic)
print("\n[*] Calculating Global SHAP Game-Theory Values (This will take a minute)...")
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_train)

# 6. Generate the Plot
print("[+] Generating Figure: Global Biomarker Signatures...")
plt.figure(figsize=(12, 10))
plt.title(f"GLOBAL SHAP Feature Importance\n(Universal Mortality Biomarkers across {X_full.shape[0]} Patients)", fontsize=16, pad=20)

# Plot the top 20 genes driving death
shap.summary_plot(shap_values, X_train, max_display=20, show=False)

plot_path = os.path.join(processed_dir, "Real_Global_SHAP_Summary.png")
plt.tight_layout()
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
print(f"\n[!!!] High-Resolution Global Plot saved to: {plot_path}")