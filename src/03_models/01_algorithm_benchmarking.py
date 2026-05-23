"""
01_algorithm_benchmarking.py

Multi-Cohort Machine Learning Benchmarking.
Evaluates baseline predictive performance across linear and non-linear algorithms.
Implements a "Mega-Train" methodology: loads the dedicated, isolated 6-cohort 
training tensor for robust 5-fold stratified cross-validation.
Utilizes cost-sensitive learning to address class imbalance without synthetic data generation.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

# Suppress warnings for clean execution logging
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
# Resolves to Multimodal_prediction_sepsis/
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"
OUT_DIR = BASE_DIR / "outputs" / "models"

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    """Main execution workflow."""
    print("[*] Initiating multi-cohort algorithm benchmarking...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD ISOLATED TRAINING TENSORS
    # ---------------------------------------------------------
    print("    -> Loading pre-isolated Mega-Train expression tensors...")
    
    # We strictly load the _train tensors. The vault is entirely bypassed.
    X_train_df = pd.read_csv(DATA_DIR / "X_train.csv.gz", compression='gzip')
    y_train_df = pd.read_csv(DATA_DIR / "y_train.csv")
    meta_train_df = pd.read_csv(DATA_DIR / "meta_train.csv")

    X_train_mega = X_train_df.values
    y_train_mega = y_train_df['Mortality'].values
    
    num_cohorts = meta_train_df['Dataset'].nunique()
    print(f"    -> Mega-Train Data: N = {len(y_train_mega)} patients ({num_cohorts} cohorts)")

    # ---------------------------------------------------------
    # 2. ALGORITHM INITIALIZATION & COST-SENSITIVE WEIGHTING
    # ---------------------------------------------------------
    # Calculate deterministic weight for XGBoost minority class
    scale_weight = float((len(y_train_mega) - y_train_mega.sum()) / (y_train_mega.sum() + 1e-9))

    models = {
        "Logistic Regression (L2)": LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42
        ),
        "Support Vector Machine": SVC(
            probability=True, class_weight='balanced', random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, class_weight='balanced', n_jobs=-1, random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=5, 
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_weight, 
            eval_metric='logloss', n_jobs=-1, random_state=42
        )
    }

    # ---------------------------------------------------------
    # 3. STRATIFIED CROSS-VALIDATION (5-FOLD)
    # ---------------------------------------------------------
    print("\n    -> Executing 5-Fold Stratified Cross-Validation...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []

    for name, model in models.items():
        print(f"       - Evaluating {name}...")
        fold_metrics = {'auc': [], 'f1': [], 'prec': [], 'rec': []}
        
        for train_idx, val_idx in skf.split(X_train_mega, y_train_mega):
            # Isolate folds
            X_fold_train, X_fold_val = X_train_mega[train_idx], X_train_mega[val_idx]
            y_fold_train, y_fold_val = y_train_mega[train_idx], y_train_mega[val_idx]
            
            # Fit and predict
            model.fit(X_fold_train, y_fold_train)
            
            y_pred = model.predict(X_fold_val)
            y_prob = model.predict_proba(X_fold_val)[:, 1]
            
            # Record metrics
            fold_metrics['auc'].append(roc_auc_score(y_fold_val, y_prob))
            fold_metrics['f1'].append(f1_score(y_fold_val, y_pred))
            fold_metrics['prec'].append(precision_score(y_fold_val, y_pred, zero_division=0))
            fold_metrics['rec'].append(recall_score(y_fold_val, y_pred, zero_division=0))
            
        # Aggregate mean performance across folds
        results.append({
            "Algorithm": name,
            "Mean_AUC": np.mean(fold_metrics['auc']),
            "Mean_F1": np.mean(fold_metrics['f1']),
            "Mean_Precision": np.mean(fold_metrics['prec']),
            "Mean_Recall": np.mean(fold_metrics['rec'])
        })

    # ---------------------------------------------------------
    # 4. EXPORT & REPORT RESULTS
    # ---------------------------------------------------------
    df_results = pd.DataFrame(results).sort_values(by="Mean_AUC", ascending=False)
    
    print("\n" + "-" * 65)
    print("MULTI-COHORT BENCHMARK RESULTS (CROSS-VALIDATION)")
    print("-" * 65)
    print(df_results.to_string(index=False))

    csv_path = OUT_DIR / "algorithm_benchmark_results.csv"
    df_results.to_csv(csv_path, index=False)
    print(f"\n[*] Benchmarking complete. Metrics exported to: {csv_path.name}")

# ==========================================
# EXECUTION GUARD
# ==========================================
if __name__ == "__main__":
    main()