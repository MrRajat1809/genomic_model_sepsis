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

print("[*] INITIATING XGBOOST BASELINE...")

base_dir = "/workspace"
data_dir = os.path.join(base_dir, "data/processed/ml_tensors")
model_out = os.path.join(base_dir, "outputs/models")
fig_out = os.path.join(base_dir, "outputs/figures")

# ADD THESE TWO LINES: Force Python to build the folders safely
os.makedirs(model_out, exist_ok=True)
os.makedirs(fig_out, exist_ok=True)

# 1. Load Data
print("    -> Loading Master Tensors (This will take a moment)...")
X = pd.read_csv(os.path.join(data_dir, "X_master.csv.gz"), compression='gzip')
y = pd.read_csv(os.path.join(data_dir, "y_master.csv"))

print(f"    -> Data loaded. X shape: {X.shape}, y shape: {y.shape}")

# 2. Check Class Imbalance
# Sepsis datasets usually have more survivors (0) than non-survivors (1)
mortality_rate = y['Mortality'].mean() * 100
print(f"    -> Mortality Rate in dataset: {mortality_rate:.1f}%")

# 3. Stratified Split
# Stratified ensures the 80/20 split keeps the exact same mortality ratio in both sets
print("    -> Splitting data (80% Train, 20% Test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y['Mortality'], test_size=0.2, stratify=y['Mortality'], random_state=42
)

# Calculate dynamic weight to handle class imbalance
scale_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9)

# 4. Initialize XGBoost
print("    -> Initializing XGBoost Classifier...")
model = xgb.XGBClassifier(
    n_estimators=300,          # Build 300 sequential trees
    learning_rate=0.05,        # Learn slowly to prevent overfitting
    max_depth=5,               # Restrict tree depth to handle noise
    subsample=0.8,             # Use 80% of patients per tree
    colsample_bytree=0.8,      # Use 80% of genes per tree
    scale_pos_weight=scale_weight, # Automatically balance the classes
    eval_metric='auc',         # Area Under the Curve (Clinical Standard)
    random_state=42,
    n_jobs=-1                  # Use all available CPU cores
)

# 5. Train
print("    -> Training Baseline Model (Hold on to your CPU)...")
model.fit(X_train, y_train)

# 6. Evaluate
print("    -> Generating Predictions...")
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1] # Get raw probabilities for AUC

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)

print("\n" + "="*50)
print("[*] BASELINE PERFORMANCE (XGBoost)")
print(f"    -> Accuracy: {acc:.4f}")
print(f"    -> ROC-AUC:  {auc:.4f}")
print("="*50)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 7. Extract Biology (Feature Importance)
print("    -> Extracting Top 20 Biological Biomarkers...")
importances = model.feature_importances_
fi_df = pd.DataFrame({'Gene': X.columns, 'Importance': importances})
fi_df = fi_df.sort_values(by='Importance', ascending=False).head(20)

# 8. Plot and Save
plt.figure(figsize=(12, 8))
sns.barplot(x='Importance', y='Gene', data=fi_df, palette='magma')
plt.title('Top 20 Sepsis Mortality Biomarkers (XGBoost)')
plt.xlabel('Relative Importance Score')
plt.ylabel('HUGO Gene Symbol')
plt.tight_layout()
plt.savefig(os.path.join(fig_out, "xgboost_baseline_importance.png"), dpi=300)
plt.close()

model.save_model(os.path.join(model_out, "xgboost_baseline.json"))
print(f"[*] Baseline Model saved to outputs/models/")
print(f"[*] Biomarker Chart saved to outputs/figures/")