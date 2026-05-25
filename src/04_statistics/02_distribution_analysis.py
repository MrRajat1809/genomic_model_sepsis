"""
02_distribution_analysis.py

Performs the Two-Sample Kolmogorov-Smirnov (KS) Test on the optimal biomarker
signature to statistically validate that the continuous probability distributions
of these features are fundamentally distinct between outcome cohorts.
Outputs a statistical registry for downstream visualization.
"""

import warnings
from pathlib import Path

import pandas as pd
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"
DEG_DIR = BASE_DIR / "data" / "processed" / "deg_tensors"
FEATURE_DIR = BASE_DIR / "outputs" / "features"

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("[*] Initiating statistical distribution analysis (KS Test)...")
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD DATA & FEATURES
    # ---------------------------------------------------------
    feature_path = FEATURE_DIR / "optimal_biomarker_panel.csv"
    if not feature_path.exists():
        print(f"[ERROR] Missing optimal feature list: {feature_path.name}")
        return

    optimal_genes = pd.read_csv(feature_path)['Optimal_Genes'].tolist()
    num_genes = len(optimal_genes)

    print("    -> Loading isolated training tensors...")
    X_train_full = pd.read_csv(DEG_DIR / "X_train_deg_subset.csv.gz", compression='gzip')
    y_train_df = pd.read_csv(DATA_DIR / "y_train.csv")

    target_col = 'Mortality' if 'Mortality' in y_train_df.columns else y_train_df.columns[0]
    y_train = y_train_df[target_col].astype(int).values

    survivors = X_train_full[y_train == 0]
    nonsurvivors = X_train_full[y_train == 1]

    print(f"       - Evaluating {num_genes} genes across {len(y_train)} patients.")

    # ---------------------------------------------------------
    # 2. STATISTICAL TESTING
    # ---------------------------------------------------------
    print("    -> Computing Two-Sample KS Tests...")
    ks_results = []
    
    for gene in optimal_genes:
        s_data = survivors[gene]
        ns_data = nonsurvivors[gene]
        
        ks_stat, p_value = ks_2samp(s_data, ns_data)
        
        ks_results.append({
            'Gene': gene,
            'KS_Statistic': ks_stat,
            'P_Value': p_value
        })

    # ---------------------------------------------------------
    # 3. EXPORT REGISTRY
    # ---------------------------------------------------------
    ks_df = pd.DataFrame(ks_results).sort_values(by='KS_Statistic', ascending=False)
    csv_path = FEATURE_DIR / f"ks_test_statistics_{num_genes}genes.csv"
    ks_df.to_csv(csv_path, index=False)

    print(f"[*] SUCCESS! Statistical registry saved to: {csv_path.name}")

if __name__ == "__main__":
    main()