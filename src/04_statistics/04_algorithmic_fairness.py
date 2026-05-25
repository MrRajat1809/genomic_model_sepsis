"""
04_algorithmic_fairness.py

Algorithmic Fairness Metrics Computation.
Merges raw demographic metadata with the pre-computed Vault predictions to 
evaluate model performance across protected attributes (Sex and Age).
Exports the stratified AUROC scores for downstream visualization.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"
RAW_META_DIR = BASE_DIR / "data" / "raw" / "geo_metadata"
METRICS_DIR = BASE_DIR / "outputs" / "metrics"

HOLDOUT_COHORT = 'GSE65682'

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def calculate_subgroup_auc(y_true, y_probs):
    if len(np.unique(y_true)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    return auc(fpr, tpr)

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("Calculating algorithmic fairness and demographic parity metrics...")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD PREDICTIONS & METADATA
    # ---------------------------------------------------------
    pred_path = METRICS_DIR / "vault_predictions.csv"
    meta_path = DATA_DIR / "meta_vault.csv"
    raw_meta_path = RAW_META_DIR / f"{HOLDOUT_COHORT}_metadata.csv"
    
    missing_files = [f.name for f in [pred_path, meta_path, raw_meta_path] if not f.exists()]
    if missing_files:
        print(f"Error: Missing files: {', '.join(missing_files)}")
        return

    preds = pd.read_csv(pred_path)
    meta_vault = pd.read_csv(meta_path)
    vault_raw_meta = pd.read_csv(raw_meta_path)

    y_vault = preds['y_true'].values
    vault_probs = preds['y_prob'].values

    # ---------------------------------------------------------
    # 2. INTEGRATE DEMOGRAPHIC VARIABLES
    # ---------------------------------------------------------
    # Ensure alignment assuming predictions match the meta_vault order
    meta_vault = pd.merge(meta_vault, vault_raw_meta, on='Patient_ID', how='left')

    if 'gender' in meta_vault.columns:
        meta_vault['Clean_Sex'] = meta_vault['gender'].astype(str).str.strip().str.title()
        meta_vault['Clean_Sex'] = meta_vault['Clean_Sex'].replace({'M': 'Male', 'F': 'Female'})
    else:
        meta_vault['Clean_Sex'] = 'Unknown'

    if 'age' in meta_vault.columns:
        meta_vault['Clean_Age'] = pd.to_numeric(meta_vault['age'], errors='coerce')
        conditions = [(meta_vault['Clean_Age'] < 60), (meta_vault['Clean_Age'] >= 60)]
        choices = ['< 60 Years', '>= 60 Years']
        meta_vault['Age_Group'] = np.select(conditions, choices, default='Unknown')
    else:
        meta_vault['Age_Group'] = 'Unknown'

    # ---------------------------------------------------------
    # 3. CALCULATE SUBGROUP METRICS
    # ---------------------------------------------------------
    results = []
    
    baseline_auc = calculate_subgroup_auc(y_vault, vault_probs)
    results.append({'Category': 'Baseline', 'Subgroup': 'Overall Cohort', 'AUROC': baseline_auc, 'N': len(y_vault)})

    for sex in ['Male', 'Female']:
        mask = (meta_vault['Clean_Sex'] == sex).values
        if mask.sum() > 0:
            val = calculate_subgroup_auc(y_vault[mask], vault_probs[mask])
            results.append({'Category': 'Biological Sex', 'Subgroup': sex, 'AUROC': val, 'N': mask.sum()})

    for age_grp in ['< 60 Years', '>= 60 Years']:
        mask = (meta_vault['Age_Group'] == age_grp).values
        if mask.sum() > 0:
            val = calculate_subgroup_auc(y_vault[mask], vault_probs[mask])
            results.append({'Category': 'Age Bracket', 'Subgroup': age_grp, 'AUROC': val, 'N': mask.sum()})

    df_results = pd.DataFrame(results).dropna()

    if len(df_results) == 1:
        print("Warning: Demographic data missing in metadata. Only baseline computed.")

    # ---------------------------------------------------------
    # 4. EXPORT
    # ---------------------------------------------------------
    out_path = METRICS_DIR / "vault_fairness_metrics.csv"
    df_results.to_csv(out_path, index=False)
    print(f"Fairness metrics exported to: {out_path.name}")

if __name__ == "__main__":
    main()