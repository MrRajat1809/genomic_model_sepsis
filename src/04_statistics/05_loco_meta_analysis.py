"""
05_loco_meta_analysis.py

Leave-One-Cohort-Out (LOCO) Meta-Analysis Computation.
Iteratively trains the XGBoost model on N-1 cohorts and tests on the held-out cohort.
Computes 1,000-iteration bootstrapped 95% CIs for each cohort and calculates 
Meta-Analysis Heterogeneity (Cochran's Q and I^2). 
Exports all metrics for downstream Forest Plot visualization.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"
DEG_DIR = BASE_DIR / "data" / "processed" / "deg_tensors"
FEATURE_DIR = BASE_DIR / "outputs" / "features"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"

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
    print("Executing LOCO validation and meta-analysis computation...")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD FEATURES & TENSORS
    # ---------------------------------------------------------
    feature_path = FEATURE_DIR / "optimal_biomarker_panel.csv"
    if not feature_path.exists():
        print(f"Error: Optimal features missing: {feature_path.name}")
        return
        
    optimal_genes = pd.read_csv(feature_path)['Optimal_Genes'].tolist()

    X_train_full = pd.read_csv(DEG_DIR / "X_train_deg_subset.csv.gz", compression='gzip')
    y_train_df = pd.read_csv(DATA_DIR / "y_train.csv")
    meta_train_df = pd.read_csv(DATA_DIR / "meta_train.csv")
    
    target_col = 'Mortality' if 'Mortality' in y_train_df.columns else y_train_df.columns[0]
    X_train_master = X_train_full[optimal_genes]
    y_train_master = y_train_df[target_col].astype(int)

    training_cohorts = meta_train_df['Dataset'].unique()
    
    forest_records = []
    roc_records = []

    # ---------------------------------------------------------
    # 2. LOCO ITERATIONS & BOOTSTRAPPING
    # ---------------------------------------------------------
    for held_out_cohort in training_cohorts:
        test_mask = meta_train_df['Dataset'] == held_out_cohort
        train_mask = ~test_mask
        
        X_train = X_train_master[train_mask]
        y_train = y_train_master[train_mask]
        X_test = X_train_master[test_mask]
        y_test = y_train_master[test_mask]
        
        if len(y_test.unique()) < 2:
            continue
            
        scale_weight = float((len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9))
        
        loco_model = SklearnCompatibleXGBClassifier(
            n_estimators=100, 
            learning_rate=0.05, 
            max_depth=4,
            scale_pos_weight=scale_weight, 
            eval_metric='logloss', 
            objective='binary:logistic',
            random_state=RANDOM_STATE, 
            n_jobs=-1
        )
        
        loco_model.fit(X_train, y_train)
        probs = loco_model.predict_proba(X_test)[:, 1]
        base_auc = roc_auc_score(y_test, probs)
        
        # Bootstrapping for CI
        rng = np.random.RandomState(RANDOM_STATE)
        boot_aucs = []
        y_test_arr = y_test.values
        
        for _ in range(N_BOOTSTRAPS):
            indices = rng.randint(0, len(y_test_arr), len(y_test_arr))
            if len(np.unique(y_test_arr[indices])) < 2:
                continue
            boot_aucs.append(roc_auc_score(y_test_arr[indices], probs[indices]))
            
        auc_lower = np.percentile(boot_aucs, 2.5)
        auc_upper = np.percentile(boot_aucs, 97.5)
        auc_se = np.std(boot_aucs)  
        
        print(f"Held Out: {held_out_cohort:<10} | AUC: {base_auc:.3f} (95% CI: [{auc_lower:.3f}, {auc_upper:.3f}])")
        
        forest_records.append({
            "Cohort": held_out_cohort,
            "N_Patients": len(y_test),
            "AUC": base_auc,
            "AUC_Lower": auc_lower,
            "AUC_Upper": auc_upper,
            "SE": auc_se
        })
        
        fpr, tpr, _ = roc_curve(y_test, probs)
        for f, t in zip(fpr, tpr):
            roc_records.append({"Cohort": held_out_cohort, "FPR": f, "TPR": t})

    # ---------------------------------------------------------
    # 3. META-ANALYSIS STATISTICS
    # ---------------------------------------------------------
    df_forest = pd.DataFrame(forest_records)
    df_forest['Weight'] = 1 / (df_forest['SE'] ** 2)
    sum_weights = df_forest['Weight'].sum()

    weighted_mean_auc = (df_forest['AUC'] * df_forest['Weight']).sum() / sum_weights

    df_forest['Q_contrib'] = df_forest['Weight'] * ((df_forest['AUC'] - weighted_mean_auc) ** 2)
    Q = df_forest['Q_contrib'].sum()
    df_stat = len(df_forest) - 1
    I2 = max(0.0, 100 * (Q - df_stat) / Q)

    se_pooled = 1 / np.sqrt(sum_weights)
    pooled_lower = weighted_mean_auc - (1.96 * se_pooled)
    pooled_upper = weighted_mean_auc + (1.96 * se_pooled)

    meta_stats = pd.DataFrame([{
        "Pooled_AUC": weighted_mean_auc,
        "Pooled_Lower": pooled_lower,
        "Pooled_Upper": pooled_upper,
        "Q": Q,
        "I2": I2,
        "df": df_stat
    }])

    # ---------------------------------------------------------
    # 4. EXPORT METRICS
    # ---------------------------------------------------------
    df_forest.to_csv(METRICS_DIR / "loco_forest_metrics.csv", index=False)
    meta_stats.to_csv(METRICS_DIR / "loco_meta_stats.csv", index=False)
    pd.DataFrame(roc_records).to_csv(METRICS_DIR / "loco_roc_curves.csv", index=False)

    print("LOCO computation and export complete.")

if __name__ == "__main__":
    main()