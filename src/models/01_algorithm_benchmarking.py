"""
01_algorithm_benchmarking.py

Multi-Cohort ML Benchmarking.
Implements the "Mega-Model" rationale: Pools 6 diverse datasets for robust 
training and cross-validation, forcing the algorithms to learn platform-agnostic 
biology. 
"""

import warnings
from pathlib import Path
import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"
FIG_OUT = BASE_DIR / "outputs" / "figures"

# The Holdout Vault (We will NOT train on this)
HOLDOUT_COHORT = 'GSE65682'

def main():
    print("[*] INITIATING ALGORITHM BENCHMARKING (MEGA-MODEL LOGIC)...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # 1. Load the Master Tensors
    print("    -> Loading Tensors...")
    X = pd.read_csv(DATA_DIR / "X_master.csv.gz", compression='gzip')
    y = pd.read_csv(DATA_DIR / "y_master.csv")
    meta = pd.read_csv(DATA_DIR / "meta_master.csv")

    # 2. The Multi-Cohort Split
    # Isolate the 6 training cohorts
    train_mask = meta['Dataset'] != HOLDOUT_COHORT
    
    X_train_mega = X[train_mask].values
    y_train_mega = y[train_mask]['Mortality'].values
    
    print(f"    -> Mega-Train Cohort: {len(y_train_mega)} patients (6 datasets)")
    print(f"    -> Locked Vault Cohort ({HOLDOUT_COHORT}): {len(y) - len(y_train_mega)} patients")

    # 3. Define Algorithms
    # Calculate scale_pos_weight for XGBoost to handle survival/death imbalance
    scale_weight = (len(y_train_mega) - y_train_mega.sum()) / (y_train_mega.sum() + 1e-9)

    models = {
        "Logistic Regression (L2)": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        "Support Vector Machine": SVC(probability=True, class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, class_weight='balanced', n_jobs=-1, random_state=42),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=150, learning_rate=0.05, max_depth=5, 
            subsample=0.8, colsample_bytree=0.8, scale_pos_weight=scale_weight, 
            eval_metric='auc', n_jobs=-1, random_state=42
        )
    }

    # 4. 5-Fold Cross Validation strictly on the Mega-Train data
    print("\n    -> Commencing 5-Fold Cross-Validation on Mega-Train Data...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    results = []

    for name, model in models.items():
        print(f"       - Evaluating {name}...")
        fold_metrics = {'auc': [], 'f1': [], 'prec': [], 'rec': []}
        
        for train_idx, val_idx in skf.split(X_train_mega, y_train_mega):
            X_fold_train, X_fold_val = X_train_mega[train_idx], X_train_mega[val_idx]
            y_fold_train, y_fold_val = y_train_mega[train_idx], y_train_mega[val_idx]
            
            model.fit(X_fold_train, y_fold_train)
            
            y_pred = model.predict(X_fold_val)
            y_prob = model.predict_proba(X_fold_val)[:, 1]
            
            fold_metrics['auc'].append(roc_auc_score(y_fold_val, y_prob))
            fold_metrics['f1'].append(f1_score(y_fold_val, y_pred))
            fold_metrics['prec'].append(precision_score(y_fold_val, y_pred, zero_division=0))
            fold_metrics['rec'].append(recall_score(y_fold_val, y_pred, zero_division=0))
            
        results.append({
            "Algorithm": name,
            "Mean_AUC": np.mean(fold_metrics['auc']),
            "Mean_F1": np.mean(fold_metrics['f1']),
            "Mean_Precision": np.mean(fold_metrics['prec']),
            "Mean_Recall": np.mean(fold_metrics['rec'])
        })

    # 5. Display Results
    df_results = pd.DataFrame(results).sort_values(by="Mean_AUC", ascending=False)
    print("\n[*] MULTI-COHORT BENCHMARK RESULTS (TRAINING PHASE):")
    print(df_results.to_string(index=False))

    # Save to CSV
    df_results.to_csv(FIG_OUT / "Algorithm_Benchmark_Results.csv", index=False)

if __name__ == "__main__":
    main()