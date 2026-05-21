"""
02_deg_visualizations.py

Generates the Gold Standard DEG Visualizations (Volcano & Heatmap).
Implements the progressive filtering color scheme:
- Background Noise (Gray)
- Robust DEGs (Red)
- Elite Biomarkers (Bright Green)
"""

import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
from statsmodels.stats.multitest import multipletests

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"
DEG_DIR = BASE_DIR / "data" / "processed" / "deg_tensors"
FIG_OUT = BASE_DIR / "outputs" / "figures"

HOLDOUT_COHORT = 'GSE65682'

def main():
    print("[*] INITIATING STORY-DRIVEN DEG VISUALIZATIONS...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # 1. Load the Data
    print("    -> Loading Master Tensors and Gene Lists...")
    X = pd.read_csv(DATA_DIR / "X_master.csv.gz", compression='gzip')
    y = pd.read_csv(DATA_DIR / "y_master.csv")
    meta = pd.read_csv(DATA_DIR / "meta_master.csv")
    
    elite_df = pd.read_csv(DEG_DIR / "Gold_Standard_Elite_DEGs.csv")
    robust_df = pd.read_csv(DEG_DIR / "All_1244_Robust_DEGs.csv")
    
    elite_genes = set(elite_df['Gene'].tolist())
    robust_genes = set(robust_df['Gene'].tolist())

    # 2. Isolate Training Cohort
    train_mask = meta['Dataset'] != HOLDOUT_COHORT
    X_train = X[train_mask]
    y_train = y[train_mask]['Mortality']
    
    survivors = X_train[y_train == 0]
    nonsurvivors = X_train[y_train == 1]

    # 3. Calculate Coordinates for the Volcano Plot
    print("    -> Calculating Volcano Coordinates...")
    p_values = []
    for gene in X.columns:
        stat, pval = ttest_ind(survivors[gene], nonsurvivors[gene], equal_var=False)
        p_values.append(pval)

    p_values = np.nan_to_num(p_values, nan=1.0) 
    _, fdr_pvals, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    
    mean_diffs = nonsurvivors.mean() - survivors.mean()

    plot_df = pd.DataFrame({
        'Gene': X.columns,
        'Mean_Diff': mean_diffs.values,
        '-Log10_FDR': -np.log10(fdr_pvals + 1e-300)
    })

    # 4. Assign Plotting Categories (Your custom logic)
    def assign_status(row):
        if row['Gene'] in elite_genes:
            return 'Elite Biomarkers (n=63)'
        elif row['Gene'] in robust_genes:
            return 'Robust DEGs (n=1244)'
        else:
            return 'Background (n=6657)'

    plot_df['Status'] = plot_df.apply(assign_status, axis=1)

    # 5. Generate Volcano Plot
    print("    -> Generating Volcano Plot...")
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="ticks")
    
    # Your requested color scheme
    palette = {
        'Elite Biomarkers (n=63)': '#00FF00',  # Bright Green
        'Robust DEGs (n=1244)': '#D62728',     # Red
        'Background (n=6657)': '#E0E0E0'       # Light Gray
    }
    
    # Sorting ensures Green is plotted on top of Red, and Red on top of Gray
    plot_df['sort_val'] = plot_df['Status'].map({
        'Background (n=6657)': 0,
        'Robust DEGs (n=1244)': 1,
        'Elite Biomarkers (n=63)': 2
    })
    plot_df = plot_df.sort_values('sort_val')

    sns.scatterplot(
        data=plot_df, x='Mean_Diff', y='-Log10_FDR', 
        hue='Status', palette=palette, 
        alpha=0.85, s=35, edgecolor='none'
    )
    
    plt.title('Volcano Plot of Sepsis Mortality DEGs (ComBat Corrected)', fontsize=16, fontweight='bold')
    plt.xlabel('Expression Shift (Z-Score Difference)', fontsize=12)
    plt.ylabel('-Log10(FDR)', fontsize=12)
    plt.legend(title='Gene Filtering Stage', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    sns.despine()
    
    plt.tight_layout()
    plt.savefig(FIG_OUT / "Fig3B_Volcano_Plot.pdf", dpi=300, bbox_inches='tight')

    # 6. Generate Heatmap of the 63 Elite Genes
    print(f"    -> Generating Heatmap for the {len(elite_genes)} Elite Genes...")
    
    sorted_indices = y_train.sort_values().index
    heatmap_data = X_train.loc[sorted_indices, list(elite_genes)].T
    
    mortality_colors = y_train.loc[sorted_indices].map({0: '#2ca02c', 1: '#d62728'})
    
    g = sns.clustermap(
        heatmap_data, cmap='vlag', center=0, 
        vmin=-2, vmax=2, 
        col_colors=mortality_colors, col_cluster=False, 
        figsize=(16, 12), yticklabels=True, xticklabels=False,
        cbar_kws={'label': 'ComBat & Z-Scored Expression'}
    )
    g.fig.suptitle(f'Expression Heatmap of the {len(elite_genes)} Elite Biomarker Genes', fontsize=18, fontweight='bold', y=1.02)
    
    plt.savefig(FIG_OUT / "Fig3A_DEG_Heatmap.pdf", dpi=300, bbox_inches='tight')

    print(f"[*] SUCCESS! Publication figures saved to {FIG_OUT.name}")

if __name__ == "__main__":
    main()