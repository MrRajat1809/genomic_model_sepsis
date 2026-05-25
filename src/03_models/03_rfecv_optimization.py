"""
03_rfecv_optimization.py

Recursive Feature Elimination with Cross-Validation (RFECV).
Utilizes XGBoost to iteratively prune the 64-gene elite DEG signature down to the 
absolute minimum number of features required to maximize predictive AUROC. 
Operates strictly on the isolated training tensor to prevent data leakage.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

# Suppress warnings for clean execution logging
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
# Resolves to Multimodal_prediction_sepsis/
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"
DEG_DIR = BASE_DIR / "data" / "processed" / "deg_tensors"
FEATURE_OUT_DIR = BASE_DIR / "outputs" / "features"
FIG_OUT = BASE_DIR / "outputs" / "figures"

# ==========================================
# COMPATIBILITY WRAPPER
# ==========================================
class SklearnCompatibleXGBClassifier(XGBClassifier):
    """
    Bypasses the Scikit-Learn 1.6.0+ RFECV tag delegation bug.
    Intercepts the Tags object and explicitly enforces the 'estimator_type' 
    to prevent pipeline crashes during feature elimination.
    """
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        if hasattr(tags, "estimator_type"):
            tags.estimator_type = "classifier"
        elif isinstance(tags, dict):
            tags["estimator_type"] = "classifier"
        return tags

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    """Main execution workflow."""
    print("[*] Initiating non-linear feature optimization (XGBoost RFECV)...")
    FEATURE_OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    
    # ---------------------------------------------------------
    # 1. LOAD ISOLATED TRAINING TENSORS
    # ---------------------------------------------------------
    print("    -> Loading isolated elite DEG training tensors...")
    
    # Strictly load the pre-filtered training subsets. Vault data is naturally excluded.
    X_train = pd.read_csv(DEG_DIR / "X_train_deg_subset.csv.gz", compression='gzip')
    y_train_df = pd.read_csv(DATA_DIR / "y_train.csv")
    
    # Enforce strict integer typing for classification targets
    target_col = 'Mortality' if 'Mortality' in y_train_df.columns else y_train_df.columns[0]
    y_train = y_train_df[target_col].astype(int).values
    
    print(f"       - Training Cohort: N = {len(y_train)} patients")
    print(f"       - Candidate Features: {X_train.shape[1]} elite genes")

    # ---------------------------------------------------------
    # 2. INITIALIZE ALGORITHM & STRATIFICATION
    # ---------------------------------------------------------
    # Deterministic calculation for cost-sensitive minority class weighting
    scale_weight = float((len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9))
    
    clf = SklearnCompatibleXGBClassifier(
        scale_pos_weight=scale_weight,
        n_estimators=100,
        learning_rate=0.05,
        max_depth=4,
        objective='binary:logistic',  
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ---------------------------------------------------------
    # 3. EXECUTE RECURSIVE FEATURE ELIMINATION
    # ---------------------------------------------------------
    print("    -> Executing RFECV (Iteratively pruning features to maximize AUROC)...")
    selector = RFECV(
        estimator=clf, 
        step=1,                
        cv=cv, 
        scoring='roc_auc',    
        min_features_to_select=1,
        n_jobs=-1
    )
    
    selector = selector.fit(X_train, y_train)
    
    optimal_k = selector.n_features_
    max_auc = selector.cv_results_['mean_test_score'].max()
    
    print(f"\n    [+] Optimization Converged!")
    print(f"       - Optimal Biomarker Panel Size : {optimal_k} genes")
    print(f"       - Maximum Cross-Val AUROC      : {max_auc:.4f}")

    # ---------------------------------------------------------
    # 4. EXPORT OPTIMAL FEATURE SET
    # ---------------------------------------------------------
    optimal_features = X_train.columns[selector.support_].tolist()
    out_csv = FEATURE_OUT_DIR / "optimal_biomarker_panel.csv"
    pd.DataFrame({'Optimal_Genes': optimal_features}).to_csv(out_csv, index=False)
    print(f"    -> Target features exported to: {out_csv.name}")

    # ---------------------------------------------------------
    # 5. GENERATE PUBLICATION FIGURE
    # ---------------------------------------------------------
    print("    -> Generating performance trajectory visualization...")
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "#333333"})
    fig, ax = plt.subplots(figsize=(8, 5))
    
    scores = selector.cv_results_['mean_test_score']
    n_features = range(1, len(scores) + 1)
    
    # Plot Trajectory
    ax.plot(n_features, scores, color='#2c7fb8', linewidth=2.5, marker='o', markersize=4)
    
    # Annotate Optimal Point
    ax.axvline(x=optimal_k, color='#e34a33', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.plot(optimal_k, max_auc, marker='*', markersize=15, color='#e34a33', markeredgecolor='black')
    
    ax.set_title('Recursive Feature Elimination (RFECV) Optimization', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Number of Genes Retained', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('Cross-Validated AUROC', fontsize=12, fontweight='bold', labelpad=10)
    
    tick_step = max(1, len(scores) // 10)
    ax.set_xticks(np.arange(0, len(scores) + tick_step, step=tick_step))
    ax.set_xlim(0, len(scores) + 1)
    
    ax.annotate(
        f'Optimal Subset: {optimal_k} Genes\nAUROC: {max_auc:.3f}', 
        xy=(optimal_k, max_auc),
        xytext=(optimal_k + (len(scores)*0.05), max_auc - 0.02),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="gray", alpha=0.9),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2", color="black")
    )
    
    sns.despine()
    out_path = FIG_OUT / "Fig_RFECV_Optimization_Curve.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 65)
    print(f"[*] Optimization sequence complete. Figure saved to {out_path.name}")
    print("=" * 65)

# ==========================================
# EXECUTION GUARD
# ==========================================
if __name__ == "__main__":
    main()