"""
02_xgboost_baseline.py

The "Molecular Geneticist" Model.
Trains a robust XGBoost baseline classifier on the high-dimensional transcriptomic
feature space (HUGO gene symbols). 

Upgrades for Publication:
- 1,000-iteration bootstrap resampling for 95% Confidence Intervals.
- Brier Score and Expected Calibration Error (ECE) for reliability.
- CSV Exporters (The R Bridge) for ROC curves and feature importance.
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, brier_score_loss, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner terminal output
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"
MODEL_OUT = BASE_DIR / "outputs" / "models"
FIG_OUT = BASE_DIR / "outputs" / "figures"
# NEW: The R-Bridge Export Directory
PLOT_DATA_OUT = BASE_DIR / "outputs" / "plot_data"

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def calculate_ece(y_true, y_prob, n_bins=10):
    """Calculates Expected Calibration Error (ECE)."""
    bins = np.linspace(0., 1., n_bins + 1)
    binids = np.digitize(y_prob, bins) - 1
    
    ece = 0.0
    for i in range(n_bins):
        bin_idx = (binids == i)
        if np.sum(bin_idx) > 0:
            bin_acc = np.mean(y_true[bin_idx])
            bin_conf = np.mean(y_prob[bin_idx])
            bin_count = np.sum(bin_idx)
            ece += (bin_count / len(y_true)) * np.abs(bin_acc - bin_conf)
    return ece

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    print("[*] INITIATING 'MOLECULAR GENETICIST' XGBOOST BASELINE...")

    # Ensure directories exist
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    PLOT_DATA_OUT.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    x_path = DATA_DIR / "X_master.csv.gz"
    y_path = DATA_DIR / "y_master.csv"

    if not x_path.exists() or not y_path.exists():
        print(f"[!] Critical Error: Tensors not found in {DATA_DIR.name}.")
        return

    print("    -> Loading Master Tensors...")
    X = pd.read_csv(x_path, compression='gzip')
    y = pd.read_csv(y_path)
    
    # 2. Stratified Split
    print("    -> Splitting data (80% Train, 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y['Mortality'], test_size=0.2, stratify=y['Mortality'], random_state=42
    )

    scale_weight = (len(y_train) - y_train.sum()) / (y_train.sum() + 1e-9)

    # 3. Train XGBoost
    print("    -> Training Baseline Model...")
    model = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_weight,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # 4. Generate Base Predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Convert y_test to numpy array for fast indexing during bootstrap
    y_test_arr = y_test.values

    # ==========================================
    # 5. THE STATISTICS ARMOR (1,000 Bootstraps)
    # ==========================================
    print("    -> Engaging Statistical Armor: 1,000 Bootstrap Resampling...")
    n_bootstraps = 1000
    boot_auc, boot_acc, boot_brier, boot_ece = [], [], [], []
    
    rng = np.random.RandomState(42) # Lock seed for reproducible bounds
    
    for i in range(n_bootstraps):
        # Sample with replacement
        indices = rng.randint(0, len(y_test_arr), len(y_test_arr))
        
        # We need both classes present to calculate AUC
        if len(np.unique(y_test_arr[indices])) < 2:
            continue
            
        y_true_b = y_test_arr[indices]
        y_prob_b = y_prob[indices]
        y_pred_b = (y_prob_b >= 0.5).astype(int)
        
        boot_auc.append(roc_auc_score(y_true_b, y_prob_b))
        boot_acc.append(accuracy_score(y_true_b, y_pred_b))
        boot_brier.append(brier_score_loss(y_true_b, y_prob_b))
        boot_ece.append(calculate_ece(y_true_b, y_prob_b))

    # Calculate 95% Confidence Intervals
    def get_ci(data):
        return np.percentile(data, 2.5), np.percentile(data, 97.5), np.median(data)

    auc_lower, auc_upper, auc_med = get_ci(boot_auc)
    brier_lower, brier_upper, brier_med = get_ci(boot_brier)
    ece_lower, ece_upper, ece_med = get_ci(boot_ece)

    print("\n" + "="*60)
    print("[*] PUBLICATION-GRADE PERFORMANCE (Molecular Geneticist)")
    print(f"    -> ROC-AUC:      {auc_med:.3f} (95% CI: [{auc_lower:.3f}, {auc_upper:.3f}])")
    print(f"    -> Brier Score:  {brier_med:.3f} (95% CI: [{brier_lower:.3f}, {brier_upper:.3f}])")
    print(f"    -> ECE:          {ece_med:.3f} (95% CI: [{ece_lower:.3f}, {ece_upper:.3f}])")
    print("="*60)

    # ==========================================
    # 6. THE R-BRIDGE EXPORTS
    # ==========================================
    print("    -> Exporting CSVs for R (ggplot2)...")
    
    # Export A: ROC Curve Data
    fpr, tpr, thresholds = roc_curve(y_test_arr, y_prob)
    roc_df = pd.DataFrame({'FPR': fpr, 'TPR': tpr, 'Threshold': thresholds, 'Modality': 'Genomic'})
    roc_df.to_csv(PLOT_DATA_OUT / "genomic_roc_curve.csv", index=False)

    # Export B: Feature Importances
    importances = model.feature_importances_
    fi_df = pd.DataFrame({'Gene': X.columns, 'Importance': importances})
    fi_df = fi_df.sort_values(by='Importance', ascending=False).head(20)
    fi_df.to_csv(PLOT_DATA_OUT / "genomic_feature_importance.csv", index=False)

    # Export C: Model Weights
    model_path = MODEL_OUT / "molecular_geneticist_baseline.json"
    model.save_model(model_path)
    
    print(f"[*] Success. All data exported to {PLOT_DATA_OUT.name}/")

if __name__ == "__main__":
    main()