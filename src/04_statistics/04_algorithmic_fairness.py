"""
04_algorithmic_fairness.py

Algorithmic Fairness and Demographic Parity Analysis.
Evaluates the clinical XGBoost model's predictive performance across protected 
demographic attributes (Biological Sex and Age). Dynamically links raw clinical 
metadata to the holdout Vault predictions to ensure the model does not exhibit 
systematic bias against vulnerable clinical subpopulations.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import auc, roc_curve

# Suppress warnings for clean execution logging
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
# Resolves to Multimodal_prediction_sepsis/
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"
RAW_META_DIR = BASE_DIR / "data" / "raw" / "geo_metadata"
FEATURE_DIR = BASE_DIR / "outputs" / "features"
MODEL_DIR = BASE_DIR / "outputs" / "models"
FIG_OUT = BASE_DIR / "outputs" / "figures"

HOLDOUT_COHORT = 'GSE65682'

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def calculate_subgroup_auc(y_true, y_probs):
    """Calculates AUROC safely, returning NaN if a subgroup lacks both classes."""
    if len(np.unique(y_true)) < 2:
        return np.nan
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    return auc(fpr, tpr)

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    """Main execution workflow."""
    print("[*] Initiating algorithmic fairness and demographic parity evaluation...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD FEATURES & TENSORS
    # ---------------------------------------------------------
    print("    -> Loading optimal features and quarantined Vault tensors...")
    
    feature_path = FEATURE_DIR / "optimal_biomarker_panel.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"[ERROR] Optimal features list missing: {feature_path.name}")
    
    optimal_genes = pd.read_csv(feature_path)['Optimal_Genes'].tolist()
    num_genes = len(optimal_genes)

    # Load pre-isolated vault data
    X_vault_full = pd.read_csv(DATA_DIR / "X_vault.csv.gz", compression='gzip')
    y_vault_df = pd.read_csv(DATA_DIR / "y_vault.csv")
    meta_vault = pd.read_csv(DATA_DIR / "meta_vault.csv")
    
    # Subset features
    X_vault = X_vault_full[optimal_genes]
    
    target_col = 'Mortality' if 'Mortality' in y_vault_df.columns else y_vault_df.columns[0]
    y_vault = y_vault_df[target_col].astype(int).values

    # ---------------------------------------------------------
    # 2. INTEGRATE RAW DEMOGRAPHIC METADATA
    # ---------------------------------------------------------
    print("    -> Extracting raw demographic variables from clinical metadata...")
    raw_meta_path = RAW_META_DIR / f"{HOLDOUT_COHORT}_metadata.csv"
    
    if not raw_meta_path.exists():
        raise FileNotFoundError(f"[ERROR] Raw metadata missing at: {raw_meta_path}")
        
    vault_raw_meta = pd.read_csv(raw_meta_path)
    meta_vault = pd.merge(meta_vault, vault_raw_meta, on='Patient_ID', how='left')

    # Standardize Biological Sex
    if 'gender' in meta_vault.columns:
        meta_vault['Clean_Sex'] = meta_vault['gender'].astype(str).str.strip().str.title()
        meta_vault['Clean_Sex'] = meta_vault['Clean_Sex'].replace({'M': 'Male', 'F': 'Female'})
    else:
        meta_vault['Clean_Sex'] = 'Unknown'

    # Standardize Age Brackets
    if 'age' in meta_vault.columns:
        meta_vault['Clean_Age'] = pd.to_numeric(meta_vault['age'], errors='coerce')
        conditions = [(meta_vault['Clean_Age'] < 60), (meta_vault['Clean_Age'] >= 60)]
        choices = ['< 60 Years', '>= 60 Years']
        meta_vault['Age_Group'] = np.select(conditions, choices, default='Unknown')
    else:
        meta_vault['Age_Group'] = 'Unknown'

    # ---------------------------------------------------------
    # 3. LOAD LOCKED CLINICAL MODEL & PREDICT
    # ---------------------------------------------------------
    model_name = f"sepsis_xgboost_{num_genes}genes_deploy_v1.json"
    model_path = MODEL_DIR / model_name
    
    print(f"    -> Loading locked XGBoost weights: {model_name}...")
    model = xgb.XGBClassifier()
    model.load_model(model_path)
    
    vault_probs = model.predict_proba(X_vault)[:, 1]
    
    # ---------------------------------------------------------
    # 4. CALCULATE SUBGROUP METRICS
    # ---------------------------------------------------------
    print("\n" + "-" * 65)
    print("FAIRNESS METRICS (AUROC BY DEMOGRAPHIC)")
    print("-" * 65)
    
    results = []

    # Overall baseline
    baseline_auc = calculate_subgroup_auc(y_vault, vault_probs)
    results.append({'Category': 'Baseline', 'Subgroup': 'Overall Cohort', 'AUROC': baseline_auc, 'N': len(y_vault)})
    print(f"    -> Overall Vault AUROC : {baseline_auc:.3f}")

    # Biological Sex Subgroups
    for sex in ['Male', 'Female']:
        mask = (meta_vault['Clean_Sex'] == sex).values
        if mask.sum() > 0:
            val = calculate_subgroup_auc(y_vault[mask], vault_probs[mask])
            results.append({'Category': 'Biological Sex', 'Subgroup': sex, 'AUROC': val, 'N': mask.sum()})
            print(f"    -> Sex: {sex:<10} (N={mask.sum():<3}) : {val:.3f}")

    # Age Bracket Subgroups
    for age_grp in ['< 60 Years', '>= 60 Years']:
        mask = (meta_vault['Age_Group'] == age_grp).values
        if mask.sum() > 0:
            val = calculate_subgroup_auc(y_vault[mask], vault_probs[mask])
            results.append({'Category': 'Age Bracket', 'Subgroup': age_grp, 'AUROC': val, 'N': mask.sum()})
            print(f"    -> Age: {age_grp:<10} (N={mask.sum():<3}) : {val:.3f}")

    df_results = pd.DataFrame(results).dropna()

    if len(df_results) == 1:
        print("\n[!] WARNING: Demographic data missing. Check GSE65682 metadata columns ('age', 'gender').")
        return

    # ---------------------------------------------------------
    # 5. GENERATE PUBLICATION FIGURE
    # ---------------------------------------------------------
    print("\n    -> Generating Algorithmic Fairness Bar Chart...")
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "#aaaaaa", "grid.color": "#eeeeee"})
    
    ax = sns.barplot(
        data=df_results, 
        y='Subgroup', 
        x='AUROC', 
        hue='Category',
        dodge=False,
        palette={"Baseline": "#333333", "Biological Sex": "#4a6fe3", "Age Bracket": "#db4325"}
    )

    # Plot baseline reference line
    plt.axvline(x=baseline_auc, color='#333333', linestyle='--', linewidth=1.5, alpha=0.7, zorder=0)
    plt.text(baseline_auc + 0.01, -0.4, f'Baseline: {baseline_auc:.3f}', color='#333333', fontsize=10, fontweight='bold')

    # Annotate bars with AUROC and N
    for i, p in enumerate(ax.patches):
        width = p.get_width()
        if width > 0:
            n_val = df_results.iloc[i]['N']
            ax.text(
                width - 0.08, p.get_y() + p.get_height()/2.,
                f'{width:.3f} (n={n_val})', 
                va='center', color='white', fontweight='bold', fontsize=10
            )

    plt.xlim(0, 1.0)
    plt.title(f'Algorithmic Fairness: {num_genes}-Gene Parity', fontsize=14, pad=15, color='#333333', fontweight='bold')
    plt.xlabel('Area Under the ROC Curve (AUROC)', fontsize=12, labelpad=10, color='#555555')
    plt.ylabel('')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    
    plt.tight_layout()
    pdf_out = FIG_OUT / f"Fig_Algorithmic_Fairness_{num_genes}Genes.pdf"
    plt.savefig(pdf_out, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[*] Fairness analysis complete. Plot saved to: {pdf_out.name}")

# ==========================================
# EXECUTION GUARD
# ==========================================
if __name__ == "__main__":
    main()