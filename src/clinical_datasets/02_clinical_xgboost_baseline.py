import os
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

print("[*] INITIATING ICU DOCTOR BASELINE (XGBoost Native Missing)...")

# Directories
base_dir = "/workspace"
data_dir = os.path.join(base_dir, "data/processed/clinical_tensors")
model_out = os.path.join(base_dir, "outputs/models")
fig_out = os.path.join(base_dir, "outputs/figures")

os.makedirs(model_out, exist_ok=True)
os.makedirs(fig_out, exist_ok=True)

# 1. Load Data
print("    -> Loading Master Clinical Matrix...")
df = pd.read_csv(os.path.join(data_dir, "clinical_master_raw.csv.gz"), compression='gzip')

# 2. Prepare Features (X) and Labels (y)
print("    -> Preparing tensors...")
# Drop the ID and the outcome to create the feature matrix
X = df.drop(columns=['Patient_ID', 'Sepsis_Outcome', 'ICU_Length_of_Stay'])
y = df['Sepsis_Outcome']

print(f"    -> Clinical Tensor Shape: {X.shape}")

# 3. Stratified Split
print("    -> Splitting data (80% Train, 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Calculate weight for the 7.3% imbalance
scale_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9)

# 4. Initialize XGBoost (Native Missing Handling is automatic for np.nan)
print("    -> Initializing Clinical XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,             # Slightly deeper trees for clinical combinations
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_weight,
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)

# 5. Train
print("    -> Training ICU Doctor (Learning from missing data naturally)...")
model.fit(X_train, y_train)

# 6. Evaluate
print("    -> Generating Predictions...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "="*50)
print("[*] ICU DOCTOR PERFORMANCE (Clinical Baseline)")
print(f"    -> Accuracy: {acc:.4f}")
print(f"    -> ROC-AUC:  {auc:.4f}")
print("="*50)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Extract Feature Importance
print("    -> Extracting Top 15 Clinical Biomarkers...")
importances = model.feature_importances_
fi_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
fi_df = fi_df.sort_values(by='Importance', ascending=False).head(15)

# 8. Plot and Save
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Feature', data=fi_df, palette='viridis')
plt.title('Top 15 Clinical Predictors of Sepsis Onset')
plt.xlabel('Relative Importance Score')
plt.ylabel('Clinical Vital/Lab')
plt.tight_layout()
plt.savefig(os.path.join(fig_out, "clinical_xgboost_importance.png"), dpi=300)
plt.close()

model.save_model(os.path.join(model_out, "icu_doctor_baseline.json"))
print(f"[*] ICU Doctor Model saved to outputs/models/")
print(f"[*] Clinical Biomarker Chart saved to outputs/figures/")