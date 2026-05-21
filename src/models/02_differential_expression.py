"""
02_differential_expression.py

Gold Standard Integrative DEG Analysis.
FIXED: Applies the 95th Percentile Elite Threshold SEPARATELY to Upregulated 
and Downregulated genes to prevent directional bias and ensure a biologically 
balanced biomarker signature.
"""

import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"
OUT_DIR = BASE_DIR / "data" / "processed" / "deg_tensors"

HOLDOUT_COHORT = 'GSE65682'
FDR_CUTOFF = 0.05
ELITE_PERCENTILE = 95  

def main():
    print("[*] INITIATING GOLD STANDARD DEG ANALYSIS (BALANCED ELITE FILTER)...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the ComBat Master Tensors
    X = pd.read_csv(DATA_DIR / "X_master.csv.gz", compression='gzip')
    y = pd.read_csv(DATA_DIR / "y_master.csv")
    meta = pd.read_csv(DATA_DIR / "meta_master.csv")

    # 2. Isolate the Mega-Train Cohort
    train_mask = meta['Dataset'] != HOLDOUT_COHORT
    X_train = X[train_mask]
    y_train = y[train_mask]['Mortality']
    meta_train = meta[train_mask]
    
    survivors = X_train[y_train == 0]
    nonsurvivors = X_train[y_train == 1]
    
    # 3. Global Statistical Testing
    p_values = []
    for gene in X.columns:
        stat, pval = ttest_ind(survivors[gene], nonsurvivors[gene], equal_var=False)
        p_values.append(pval)

    p_values = np.nan_to_num(p_values, nan=1.0) 
    reject, fdr_pvals, _, _ = multipletests(p_values, alpha=FDR_CUTOFF, method='fdr_bh')

    stats_df = pd.DataFrame({
        'Gene': X.columns,
        'FDR': fdr_pvals,
        'Significant': reject
    })

    # 4. Calculate Global Effect Size & Composite Score
    mean_diffs = nonsurvivors.mean() - survivors.mean()
    stats_df['Mean_Diff'] = stats_df['Gene'].map(mean_diffs)
    stats_df['Composite_Score'] = stats_df['Mean_Diff'].abs() * -np.log10(stats_df['FDR'] + 1e-300)

    # 5. Cohort Consistency Matrix
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
    stats_df['Max_Agreement'] = np.maximum(stats_df['Pos_Votes'], stats_df['Neg_Votes'])
    stats_df['Direction'] = np.where(stats_df['Pos_Votes'] > stats_df['Neg_Votes'], 'Upregulated', 'Downregulated')

    # 6. Apply The Base Filter
    robust_genes = stats_df[
        (stats_df['Significant'] == True) & 
        (stats_df['Max_Agreement'] >= 5)
    ]

    up_count = (robust_genes['Direction'] == 'Upregulated').sum()
    down_count = (robust_genes['Direction'] == 'Downregulated').sum()

    print(f"\n[*] BASE FILTERING COMPLETE:")
    print(f"    -> Survived FDR < {FDR_CUTOFF}: {stats_df['Significant'].sum()} genes")
    print(f"    -> Survived Cohort Consistency (>=5/6): {len(robust_genes)} true universal biomarkers")

    # =====================================================================
    # 7. THE FIX: Independent "Elite" Filtering for Up and Down Genes
    # =====================================================================
    up_genes = robust_genes[robust_genes['Direction'] == 'Upregulated']
    down_genes = robust_genes[robust_genes['Direction'] == 'Downregulated']

    up_thresh = np.percentile(up_genes['Composite_Score'], ELITE_PERCENTILE)
    down_thresh = np.percentile(down_genes['Composite_Score'], ELITE_PERCENTILE)

    elite_up = up_genes[up_genes['Composite_Score'] >= up_thresh]
    elite_down = down_genes[down_genes['Composite_Score'] >= down_thresh]
    
    # Recombine them into our final elite signature
    elite_genes = pd.concat([elite_up, elite_down]).sort_values(by='Composite_Score', ascending=False)

    print(f"\n[*] BALANCED ELITE FILTERING ({ELITE_PERCENTILE}th Percentile Separately):")
    print(f"    -> Upregulated Threshold: {up_thresh:.2f} | Genes Passed: {len(elite_up)}")
    print(f"    -> Downregulated Threshold: {down_thresh:.2f} | Genes Passed: {len(elite_down)}")
    print(f"    -> Total Elite Genes Secured: {len(elite_genes)}")

    # 8. Save the Output
    final_genes_list = elite_genes['Gene'].tolist()
    X_deg = X[final_genes_list] 
    
    X_deg.to_csv(OUT_DIR / "X_deg_master.csv.gz", index=False, compression='gzip')
    elite_genes.to_csv(OUT_DIR / "Gold_Standard_Elite_DEGs.csv", index=False)
    robust_genes.to_csv(OUT_DIR / "All_1244_Robust_DEGs.csv", index=False)

    print(f"\n[*] SUCCESS! Saved balanced DEG tensors to {OUT_DIR.name}")

if __name__ == "__main__":
    main()