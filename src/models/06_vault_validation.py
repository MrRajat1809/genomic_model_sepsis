"""
06_vault_vaildation.py

The Final Clinical Deployment Phase.
Stripped of over-engineering: uses the optimal 26 genes and robust, 
regularized hyperparameters to maximize unseen Vault generalization.
"""

import warnings
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"
DEG_DIR = BASE_DIR / "data" / "processed" / "deg_tensors"
PLOT_DATA_DIR = BASE_DIR / "outputs" / "plot_data"
FIG_OUT = BASE_DIR / "outputs" / "figures"
MODEL_OUT = BASE_DIR / "outputs" / "models"

HOLDOUT_COHORT = 'GSE65682'

def main():
    print("[*] INITIATING DEPLOYMENT-READY CLINICAL VALIDATION...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    
    # 1. Load the Optimal 26 Genes
    optimal_features_path = PLOT_DATA_DIR / "03_optimal_feature_list.csv"
    if not optimal_features_path.exists():
        raise FileNotFoundError("Run 04_rfecv_optimization.py first!")
    
    optimal_genes = pd.read_csv(optimal_features_path)['Optimal_Genes'].tolist()
    print(f"    -> Loaded {len(optimal_genes)} optimal genes from RFECV.")

    # 2. Load and Subset Tensors
    X_master = pd.read_csv(DEG_DIR / "X_deg_master.csv.gz", compression='gzip')
    y_master = pd.read_csv(DATA_DIR / "y_master.csv")
    meta = pd.read_csv(DATA_DIR / "meta_master.csv")
    
    X_optimal = X_master[optimal_genes]
    target_col = 'Mortality' if 'Mortality' in y_master.columns else y_master.columns[0]
    
    # 3. Split into Train and Vault
    train_mask = meta['Dataset'] != HOLDOUT_COHORT
    vault_mask = meta['Dataset'] == HOLDOUT_COHORT
    
    X_train = X_optimal[train_mask]
    y_train = y_master[train_mask][target_col].astype(int)
    
    X_vault = X_optimal[vault_mask]
    y_vault = y_master[vault_mask][target_col].astype(int)
    
    print(f"    -> Training shape: {X_train.shape}")
    print(f"    -> Vault (Blind) shape: {X_vault.shape}")

    # 4. Train Robust Base Model (No Overfitting Grid Search)
    print("    -> Training regularized XGBoost architecture...")
    scale_weight = float((len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9))
    
    final_model = XGBClassifier(
        scale_pos_weight=scale_weight,
        n_estimators=100,      # Regularized to prevent noise memorization
        max_depth=4,           # Shallow trees for robustness
        learning_rate=0.05,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    final_model.fit(X_train, y_train)

    # 5. Save the Model Weights
    model_path = MODEL_OUT / "sepsis_xgboost_26genes_deploy_v1.json"
    final_model.save_model(model_path)
    print(f"    -> [SAVED] Locked model weights written to {model_path.name}")

    # 6. Predict on Unseen Vault
    print("\n    -> Unleashing Model on the Unseen Vault (Standard Calibration)...")
    vault_probs = final_model.predict_proba(X_vault)[:, 1]
    y_pred_class = final_model.predict(X_vault) # Uses the natural 0.5 threshold shifted by scale_pos_weight

    # 7. Evaluate Performance
    vault_auc = roc_auc_score(y_vault, vault_probs)
    print(f"\n========================================")
    print(f"[*] FINAL VAULT COHORT AUROC: {vault_auc:.4f}")
    print(f"========================================\n")
    
    print("[*] Classification Report:")
    print(classification_report(y_vault, y_pred_class, target_names=["Survivor", "Non-Survivor"]))

    # 8. Plot Final ROC Curve
    fpr, tpr, _ = roc_curve(y_vault, vault_probs)
    
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "#333333"})
    fig, ax = plt.subplots(figsize=(7, 7))
    
    ax.plot(fpr, tpr, color='#e34a33', linewidth=2.5, label=f'Deploy Model (AUC = {vault_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5)
    
    ax.set_title('Final Clinical Validation (GSE65682)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12, frameon=True, edgecolor='black')
    
    out_path = FIG_OUT / "Fig6_Deploy_Vault_ROC.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[*] SUCCESS! Final ROC curve saved to: {out_path.name}")

if __name__ == "__main__":
    main()