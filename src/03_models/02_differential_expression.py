"""
02_differential_expression.py

Statistical Differential Expression Gene (DEG) Analysis.
Reduces feature dimensionality prior to machine learning optimization.
Performs the following operations on the strictly isolated training tensor:
1. Two-sided independent T-tests with Benjamini-Hochberg FDR correction (alpha=0.05).
2. Cross-cohort consistency filtering (requires identical directional shift in >= 5/6 cohorts).
3. Stratified percentile filtering (95th percentile) applied independently to 
   upregulated and downregulated genes to ensure a biologically balanced signature.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

# Suppress warnings for clean execution logging
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
# Resolves to Multimodal_prediction_sepsis/
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"
OUT_DIR = BASE_DIR / "data" / "processed" / "deg_tensors"

FDR_CUTOFF = 0.05
PERCENTILE_CUTOFF = 95  

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    """Main execution workflow."""
    print("[*] Initiating statistical DEG analysis and feature reduction...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD STRICTLY ISOLATED TRAINING DATA
    # ---------------------------------------------------------
    print("    -> Loading isolated Mega-Train expression tensors...")
    X_train = pd.read_csv(DATA_DIR / "X_train.csv.gz", compression='gzip')
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")['Mortality']
    meta_train = pd.read_csv(DATA_DIR / "meta_train.csv")
    
    survivors = X_train[y_train.values == 0]
    nonsurvivors = X_train[y_train.values == 1]
    
    print(f"       - Training Cohort: N = {len(y_train)} patients")
    print(f"       - Total Input Features: {X_train.shape[1]} genes")

    # ---------------------------------------------------------
    # 2. GLOBAL STATISTICAL TESTING & FDR CORRECTION
    # ---------------------------------------------------------
    print("\n    -> Executing independent T-tests and Benjamini-Hochberg correction...")
    p_values = []
    
    for gene in X_train.columns:
        # equal_var=False enforces Welch's t-test (robust to unequal variances)
        stat, pval = ttest_ind(survivors[gene], nonsurvivors[gene], equal_var=False)
        p_values.append(pval)

    p_values = np.nan_to_num(p_values, nan=1.0) 
    reject, fdr_pvals, _, _ = multipletests(p_values, alpha=FDR_CUTOFF, method='fdr_bh')

    stats_df = pd.DataFrame({
        'Gene': X_train.columns,
        'FDR_pvalue': fdr_pvals,
        'Significant': reject
    })

    # ---------------------------------------------------------
    # 3. EFFECT SIZE & COMPOSITE SCORE CALCULATION
    # ---------------------------------------------------------
    # Positive Mean_Diff = Upregulated in Non-Survivors
    mean_diffs = nonsurvivors.mean() - survivors.mean()
    stats_df['Mean_Diff'] = stats_df['Gene'].map(mean_diffs)
    
    # Composite Score penalizes low effect sizes and rewards high statistical significance
    stats_df['Composite_Score'] = stats_df['Mean_Diff'].abs() * -np.log10(stats_df['FDR_pvalue'] + 1e-300)

    # ---------------------------------------------------------
    # 4. CROSS-COHORT CONSISTENCY FILTERING
    # ---------------------------------------------------------
    print("    -> Evaluating cross-cohort directional consistency...")
    training_cohorts = meta_train['Dataset'].unique()
    shifts_per_cohort = {}
    
    for cohort in training_cohorts:
        mask = (meta_train['Dataset'] == cohort)
        c_surv = X_train[mask & (y_train == 0)]
        c_dead = X_train[mask & (y_train == 1)]
        shifts_per_cohort[cohort] = c_dead.mean() - c_surv.mean()
        
    shifts_df = pd.DataFrame(shifts_per_cohort)
    stats_df['Pos_Votes'] = (shifts_df > 0).sum(axis=1).values
    stats_df['Neg_Votes'] = (shifts_df < 0).sum(axis=1).values
    
    # Enforce a strict majority agreement (>= 5 out of 6 cohorts)
    stats_df['Max_Agreement'] = np.maximum(stats_df['Pos_Votes'], stats_df['Neg_Votes'])
    stats_df['Direction'] = np.where(stats_df['Pos_Votes'] > stats_df['Neg_Votes'], 'Upregulated', 'Downregulated')

    robust_genes = stats_df[
        (stats_df['Significant'] == True) & 
        (stats_df['Max_Agreement'] >= 5)
    ]

    print(f"       - Survived FDR < {FDR_CUTOFF}: {stats_df['Significant'].sum()} genes")
    print(f"       - Survived Cohort Consistency (>=5/6): {len(robust_genes)} robust genes")

    # ---------------------------------------------------------
    # 5. STRATIFIED PERCENTILE FILTERING (ELITE SUBSET)
    # ---------------------------------------------------------
    print(f"\n    -> Applying top {PERCENTILE_CUTOFF}th percentile stratification threshold...")
    up_genes = robust_genes[robust_genes['Direction'] == 'Upregulated']
    down_genes = robust_genes[robust_genes['Direction'] == 'Downregulated']

    up_thresh = np.percentile(up_genes['Composite_Score'], PERCENTILE_CUTOFF)
    down_thresh = np.percentile(down_genes['Composite_Score'], PERCENTILE_CUTOFF)

    elite_up = up_genes[up_genes['Composite_Score'] >= up_thresh]
    elite_down = down_genes[down_genes['Composite_Score'] >= down_thresh]
    
    elite_genes = pd.concat([elite_up, elite_down]).sort_values(by='Composite_Score', ascending=False)

    print(f"       - Upregulated Threshold: {up_thresh:.2f} | Yield: {len(elite_up)} genes")
    print(f"       - Downregulated Threshold: {down_thresh:.2f} | Yield: {len(elite_down)} genes")
    print(f"       [+] Total Sub-Selected Features: {len(elite_genes)} genes")

    # ---------------------------------------------------------
    # 6. TENSOR SLICING & EXPORT
    # ---------------------------------------------------------
    print("\n    -> Exporting reduced dimensionality tensors and statistical reports...")
    
    # Slice ONLY the training tensor. The vault remains untouched.
    final_genes_list = elite_genes['Gene'].tolist()
    X_train_deg = X_train[final_genes_list] 
    
    # Define professional export paths
    tensor_out = OUT_DIR / "X_train_deg_subset.csv.gz"
    elite_out = OUT_DIR / "deg_top_percentile_features.csv"
    robust_out = OUT_DIR / "deg_full_robust_features.csv"
    
    X_train_deg.to_csv(tensor_out, index=False, compression='gzip')
    elite_genes.to_csv(elite_out, index=False)
    robust_genes.to_csv(robust_out, index=False)

    print(f"       - Reduced Training Tensor exported to : {tensor_out.name}")
    print(f"       - Elite Features list exported to     : {elite_out.name}")
    print(f"       - Full Robust Features list exported to: {robust_out.name}")
    
    print("\n" + "=" * 65)
    print(f"[*] Differential Expression Analysis Complete.")
    print("=" * 65)

# ==========================================
# EXECUTION GUARD
# ==========================================
if __name__ == "__main__":
    main()