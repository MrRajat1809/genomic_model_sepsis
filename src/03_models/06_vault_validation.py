"""
06_vault_validation.py

The Final Clinical Deployment and External Validation Phase.
Trains the final optimized XGBoost architecture on the 6-cohort Mega-Train tensor 
using the RFECV-derived optimal biomarker panel. Deploys the locked model onto 
the strictly quarantined, un-ComBatted GSE65682 holdout vault to establish the 
true, un-leaked external generalization AUROC.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from xgboost import XGBClassifier

# Suppress warnings for clean execution logging
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
# Resolves to Multimodal_prediction_sepsis/
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"
FEATURE_DIR = BASE_DIR / "outputs" / "features"
FIG_OUT = BASE_DIR / "outputs" / "figures"
MODEL_OUT = BASE_DIR / "outputs" / "models"

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    """Main execution workflow."""
    print("[*] Initiating final clinical deployment and external validation...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. LOAD OPTIMAL BIOMARKER PANEL
    # ---------------------------------------------------------
    optimal_features_path = FEATURE_DIR / "optimal_biomarker_panel.csv"
    if not optimal_features_path.exists():
        raise FileNotFoundError(f"[ERROR] RFECV optimization file missing: {optimal_features_path.name}")
    
    optimal_genes = pd.read_csv(optimal_features_path)['Optimal_Genes'].tolist()
    num_genes = len(optimal_genes)
    print(f"    -> Loaded optimal biomarker signature: {num_genes} features.")

    # ---------------------------------------------------------
    # 2. LOAD PRE-ISOLATED TENSORS
    # ---------------------------------------------------------
    print("    -> Loading isolated training and quarantined vault tensors...")
    
    # Load Training Data
    X_train_full = pd.read_csv(DATA_DIR / "X_train.csv.gz", compression='gzip')
    y_train_df = pd.read_csv(DATA_DIR / "y_train.csv")
    
    # Load Vault Data (GSE65682 - Strictly Z-Scored, Bypassed ComBat)
    X_vault_full = pd.read_csv(DATA_DIR / "X_vault.csv.gz", compression='gzip')
    y_vault_df = pd.read_csv(DATA_DIR / "y_vault.csv")

    # Subset to the optimal features
    X_train = X_train_full[optimal_genes]
    X_vault = X_vault_full[optimal_genes]
    
    target_col = 'Mortality' if 'Mortality' in y_train_df.columns else y_train_df.columns[0]
    
    y_train = y_train_df[target_col].astype(int)
    y_vault = y_vault_df[target_col].astype(int)
    
    print(f"       - Mega-Train shape: {X_train.shape[0]} patients, {X_train.shape[1]} genes")
    print(f"       - Blind Vault shape: {X_vault.shape[0]} patients, {X_vault.shape[1]} genes")

    # ---------------------------------------------------------
    # 3. TRAIN FINAL REGULARIZED MODEL
    # ---------------------------------------------------------
    print("\n    -> Training regularized XGBoost deployment architecture...")
    scale_weight = float((len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9))
    
    final_model = XGBClassifier(
        scale_pos_weight=scale_weight,
        n_estimators=100,      # Regularized to prevent noise memorization
        max_depth=4,           # Shallow trees for maximal robustness
        learning_rate=0.05,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    final_model.fit(X_train, y_train)

    # Export locked model weights
    model_path = MODEL_OUT / f"sepsis_xgboost_{num_genes}genes_deploy_v1.json"
    final_model.save_model(model_path)
    print(f"       [+] Model weights locked and saved to: {model_path.name}")

    # ---------------------------------------------------------
    # 4. EXTERNAL VALIDATION (THE ULTIMATE TEST)
    # ---------------------------------------------------------
    print("\n    -> Deploying model onto the completely unseen, un-harmonized Vault...")
    vault_probs = final_model.predict_proba(X_vault)[:, 1]
    y_pred_class = final_model.predict(X_vault)

    vault_auc = roc_auc_score(y_vault, vault_probs)
    
    print("\n" + "=" * 65)
    print(f"[*] FINAL EXTERNAL VALIDATION (VAULT) AUROC: {vault_auc:.4f}")
    print("=" * 65)
    
    print("\n[*] Full Classification Report:")
    print("-" * 65)
    print(classification_report(y_vault, y_pred_class, target_names=["Survivor", "Non-Survivor"]))

    # ---------------------------------------------------------
    # 5. GENERATE PUBLICATION FIGURE
    # ---------------------------------------------------------
    print("\n    -> Plotting final external validation ROC curve...")
    fpr, tpr, _ = roc_curve(y_vault, vault_probs)
    
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "#333333"})
    fig, ax = plt.subplots(figsize=(7, 7))
    
    ax.plot(fpr, tpr, color='#e34a33', linewidth=2.5, label=f'External Validation (AUC = {vault_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=1.5)
    
    ax.set_title(f'Final Clinical Validation: {num_genes}-Gene Signature', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=12, frameon=True, edgecolor='black')
    
    sns.despine()
    out_path = FIG_OUT / "Fig_Final_Deploy_Vault_ROC.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[*] Execution complete. Figure saved to: {out_path.name}")

# ==========================================
# EXECUTION GUARD
# ==========================================
if __name__ == "__main__":
    main()