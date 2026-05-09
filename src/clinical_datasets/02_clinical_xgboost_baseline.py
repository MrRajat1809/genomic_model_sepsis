"""
02_clinical_xgboost_baseline.py

The "ICU Doctor" Model.
Trains a robust XGBoost baseline classifier on the clinical feature space (vitals and labs)
to predict sepsis onset. Leverages XGBoost's native sparsity awareness to handle missing 
clinical data without explicit imputation. Extracts feature importances and saves the 
trained weights for downstream meta-model integration.
"""

import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner terminal output
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
# Dynamically resolve paths from the script's location (src/clinical_datasets/)
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "clinical_tensors"
MODEL_OUT = BASE_DIR / "outputs" / "models"
FIG_OUT = BASE_DIR / "outputs" / "figures"

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    print("[*] INITIATING ICU DOCTOR BASELINE (XGBoost Native Missing)...")

    # Force Python to build the folders safely
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    clinical_path = DATA_DIR / "clinical_master_raw.csv.gz"
    
    if not clinical_path.exists():
        print(f"[!] Critical Error: Clinical tensor not found at {clinical_path}")
        print("[!] Please run the clinical data parser script first.")
        return

    print("    -> Loading Master Clinical Matrix...")
    df = pd.read_csv(clinical_path, compression='gzip')

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

    # Calculate weight for the class imbalance
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
    
    fig_path = FIG_OUT / "clinical_xgboost_importance.png"
    plt.savefig(fig_path, dpi=300)
    plt.close()

    model_path = MODEL_OUT / "icu_doctor_baseline.json"
    model.save_model(model_path)
    
    print(f"\n[*] ICU Doctor Model saved to: {model_path}")
    print(f"[*] Clinical Biomarker Chart saved to: {fig_path}")

if __name__ == "__main__":
    main()