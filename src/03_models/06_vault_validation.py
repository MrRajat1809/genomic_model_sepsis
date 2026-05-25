"""
06_vault_validation.py

Clinical Deployment and External Validation Phase.
1. Performs 5-Fold Stratified Cross-Validation on the training data.
2. Trains the final XGBoost classifier on the complete training tensor.
3. Deploys the model onto the un-harmonized GSE65682 holdout vault.
4. Performs 1,000-iteration bootstrapping to calculate 95% Confidence Intervals.
Exports predictions and pre-calculated ROC boundaries for visualization.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"
DEG_DIR = BASE_DIR / "data" / "processed" / "deg_tensors"
FEATURE_DIR = BASE_DIR / "outputs" / "features"
MODEL_OUT = BASE_DIR / "outputs" / "models"
METRICS_OUT = BASE_DIR / "outputs" / "metrics"

N_BOOTSTRAPS = 1000
RANDOM_STATE = 42

# ==========================================
# COMPATIBILITY WRAPPER
# ==========================================
class SklearnCompatibleXGBClassifier(xgb.XGBClassifier):
    """Bypasses Scikit-Learn 1.6.0+ tag delegation limitations."""
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        if hasattr(tags, "estimator_type"):
            tags.estimator_type = "classifier"
        elif isinstance(tags, dict):
            tags["estimator_type"] = "classifier"
        return tags

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("Executing clinical deployment and validation pipeline...")
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    METRICS_OUT.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. LOAD FEATURES AND TENSORS
    # ---------------------------------------------------------
    feature_path = FEATURE_DIR / "optimal_biomarker_panel.csv"
    if not feature_path.exists():
        print(f"Error: Feature list missing at {feature_path}")
        return
    
    optimal_genes = pd.read_csv(feature_path)['Optimal_Genes'].tolist()
    num_genes = len(optimal_genes)

    print("Loading datasets...")
    X_train_full = pd.read_csv(DEG_DIR / "X_train_deg_subset.csv.gz", compression='gzip')
    y_train_df = pd.read_csv(DATA_DIR / "y_train.csv")
    
    X_vault_full = pd.read_csv(DATA_DIR / "X_vault.csv.gz", compression='gzip')
    y_vault_df = pd.read_csv(DATA_DIR / "y_vault.csv")

    X_train = X_train_full[optimal_genes]
    X_vault = X_vault_full[optimal_genes]
    
    target_col = 'Mortality' if 'Mortality' in y_train_df.columns else y_train_df.columns[0]
    y_train = y_train_df[target_col].astype(int)
    y_vault = y_vault_df[target_col].astype(int)

    # ---------------------------------------------------------
    # 2. INTERNAL VALIDATION (5-FOLD CV)
    # ---------------------------------------------------------
    print("Executing 5-Fold Stratified CV...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        scale_weight_cv = float((len(y_tr) - y_tr.sum()) / (y_tr.sum() + 1e-9))
        
        model_cv = SklearnCompatibleXGBClassifier(
            scale_pos_weight=scale_weight_cv,
            n_estimators=100, max_depth=4, learning_rate=0.05,
            objective='binary:logistic', eval_metric='logloss',
            random_state=RANDOM_STATE, n_jobs=-1
        )
        model_cv.fit(X_tr, y_tr)
        val_probs = model_cv.predict_proba(X_val)[:, 1]
        
        for yt, yp in zip(y_val, val_probs):
            cv_results.append({'Fold': fold, 'y_true': yt, 'y_prob': yp})
            
    cv_df = pd.DataFrame(cv_results)
    cv_df.to_csv(METRICS_OUT / "cv_fold_predictions.csv", index=False)

    # ---------------------------------------------------------
    # 3. DEPLOYMENT MODEL TRAINING
    # ---------------------------------------------------------
    print("Training final model on full training set...")
    scale_weight = float((len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9))
    
    final_model = SklearnCompatibleXGBClassifier(
        scale_pos_weight=scale_weight,
        n_estimators=100,      
        max_depth=4,           
        learning_rate=0.05,
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    final_model.fit(X_train, y_train)

    model_path = MODEL_OUT / f"sepsis_xgboost_{num_genes}genes.json"
    final_model.save_model(model_path)

    # ---------------------------------------------------------
    # 4. EXTERNAL VALIDATION (VAULT)
    # ---------------------------------------------------------
    print("Evaluating on external holdout vault...")
    vault_probs = final_model.predict_proba(X_vault)[:, 1]
    y_pred_class = final_model.predict(X_vault)

    vault_auc = roc_auc_score(y_vault, vault_probs)
    print(f"Vault AUROC: {vault_auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_vault, y_pred_class, target_names=["Survivor", "Non-Survivor"]))

    pd.DataFrame({
        'y_true': y_vault,
        'y_prob': vault_probs,
        'y_pred': y_pred_class
    }).to_csv(METRICS_OUT / "vault_predictions.csv", index=False)

    # ---------------------------------------------------------
    # 5. BOOTSTRAP ANALYSIS (95% CI)
    # ---------------------------------------------------------
    print("Computing 95% Confidence Intervals via bootstrapping...")
    rng = np.random.default_rng(RANDOM_STATE)
    indices = np.arange(len(y_vault))
    
    bootstrapped_aucs = []
    bootstrapped_tprs = []
    mean_fpr = np.linspace(0, 1, 100)

    for _ in range(N_BOOTSTRAPS):
        boot_idx = rng.choice(indices, size=len(indices), replace=True)
        y_true_boot = y_vault.iloc[boot_idx]
        y_prob_boot = vault_probs[boot_idx]

        # Skip iterations that sample only one class
        if len(np.unique(y_true_boot)) < 2:
            continue

        fpr_boot, tpr_boot, _ = roc_curve(y_true_boot, y_prob_boot)
        bootstrapped_aucs.append(roc_auc_score(y_true_boot, y_prob_boot))

        interp_tpr = np.interp(mean_fpr, fpr_boot, tpr_boot)
        interp_tpr[0] = 0.0
        bootstrapped_tprs.append(interp_tpr)

    auc_lower = np.percentile(bootstrapped_aucs, 2.5)
    auc_upper = np.percentile(bootstrapped_aucs, 97.5)
    
    tpr_lower = np.percentile(bootstrapped_tprs, 2.5, axis=0)
    tpr_upper = np.percentile(bootstrapped_tprs, 97.5, axis=0)
    tpr_mean = np.mean(bootstrapped_tprs, axis=0)
    
    print(f"Vault AUROC 95% CI: [{auc_lower:.4f} - {auc_upper:.4f}]")

    pd.DataFrame({
        'fpr': mean_fpr,
        'tpr_mean': tpr_mean,
        'tpr_lower': tpr_lower,
        'tpr_upper': tpr_upper
    }).to_csv(METRICS_OUT / "vault_roc_bounds.csv", index=False)

    print("Pipeline execution complete.")

if __name__ == "__main__":
    main()