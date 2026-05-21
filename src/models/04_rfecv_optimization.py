"""
04_rfecv_optimization.py

Performs Recursive Feature Elimination with Cross-Validation (RFECV) using XGBoost.
Identifies the absolute minimum number of genes required from the 63-DEG elite pool.
"""

import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"
DEG_DIR = BASE_DIR / "data" / "processed" / "deg_tensors"
PLOT_DATA_DIR = BASE_DIR / "outputs" / "plot_data"
FIG_OUT = BASE_DIR / "outputs" / "figures"

HOLDOUT_COHORT = 'GSE65682'

# ==========================================
# [THE BULLETPROOF FIX]: Intercept and Mutate Tags
# ==========================================
class SklearnCompatibleXGBClassifier(XGBClassifier):
    """
    Bypasses the Scikit-Learn 1.6.0+ RFECV tag delegation bug.
    Instead of disabling the tags, we intercept the Tags object
    and explicitly force the 'estimator_type' to be 'classifier'
    so RFECV accepts it without crashing.
    """
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        # Handle the new Scikit-Learn 1.6 Tags dataclass
        if hasattr(tags, "estimator_type"):
            tags.estimator_type = "classifier"
        # Fallback just in case an older dict format is returned
        elif isinstance(tags, dict):
            tags["estimator_type"] = "classifier"
        return tags


def main():
    print("[*] INITIATING XGBOOST RFECV OPTIMIZATION...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    PLOT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Load the ML Tensors
    print("    -> Loading aligned tensors...")
    X_elite = pd.read_csv(DEG_DIR / "X_deg_master.csv.gz", compression='gzip')
    y = pd.read_csv(DATA_DIR / "y_master.csv")
    meta = pd.read_csv(DATA_DIR / "meta_master.csv")
    
    # 2. Split into Train and Vault
    train_mask = meta['Dataset'] != HOLDOUT_COHORT
    X_train = X_elite[train_mask]
    
    target_col = 'Mortality' if 'Mortality' in y.columns else y.columns[0]
    
    # Force y_train to be strict integers
    y_train = y[train_mask][target_col].astype(int)
    
    print(f"    -> Secured {X_train.shape[0]} training patients and {X_train.shape[1]} candidate genes.")
    print(f"    -> Vault cohort ({HOLDOUT_COHORT}) safely excluded from optimization.")

    # 3. Setup Classifier using the Patched Wrapper
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

    # 4. Execute RFECV
    print("    -> Running Recursive Feature Elimination (this will take a minute)...")
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
    
    print(f"\n[*] OPTIMIZATION COMPLETE!")
    print(f"    -> Optimal Number of Genes : {optimal_k}")
    print(f"    -> Maximum Cross-Val AUROC : {max_auc:.4f}")

    # 5. Save the list of optimal features
    optimal_features = X_train.columns[selector.support_].tolist()
    out_csv = PLOT_DATA_DIR / "03_optimal_feature_list.csv"
    pd.DataFrame({'Optimal_Genes': optimal_features}).to_csv(out_csv, index=False)

    # 6. Generate Curve
    print("    -> Plotting performance curve...")
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "#333333"})
    fig, ax = plt.subplots(figsize=(8, 5))
    
    scores = selector.cv_results_['mean_test_score']
    n_features = range(1, len(scores) + 1)
    
    ax.plot(n_features, scores, color='#2c7fb8', linewidth=2.5, marker='o', markersize=4)
    ax.axvline(x=optimal_k, color='#e34a33', linestyle='--', linewidth=1.5, alpha=0.8)
    ax.plot(optimal_k, max_auc, marker='*', markersize=15, color='#e34a33', markeredgecolor='black')
    
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
    
    out_path = FIG_OUT / "Fig5_RFECV_Optimization_Curve.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[*] SUCCESS! RFECV plot saved to: {out_path.name}")
    print(f"[*] SUCCESS! Optimal gene list saved to: {out_csv.name}")

if __name__ == "__main__":
    main()