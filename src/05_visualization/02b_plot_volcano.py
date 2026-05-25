"""
02b_plot_volcano.py

Generates Panel B: Annotated Volcano Plot.
Dynamically recalculates global training statistics to construct the background 
landscape, explicitly highlighting the progressive dimensionality reduction stages.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
FEATURE_DIR = BASE_DIR / "outputs" / "features"
FIG_OUT = BASE_DIR / "outputs" / "figures"

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("[*] Generating Figure 3 Panel B: Annotated Volcano Plot...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD MASTER TENSORS & FILTERED GENE LISTS
    # ---------------------------------------------------------
    print("    -> Loading full training tensors and filtered feature lists...")
    
    X_train = pd.read_csv(DATA_DIR / "X_train.csv.gz", compression='gzip')
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")
    target_col = 'Mortality' if 'Mortality' in y_train.columns else y_train.columns[0]
    
    robust_path = DEG_DIR / "deg_full_robust_features.csv"
    feature_path = FEATURE_DIR / "optimal_biomarker_panel.csv"
    
    if not robust_path.exists() or not feature_path.exists():
        print(f"[ERROR] Missing required feature lists in {DEG_DIR.name} or {FEATURE_DIR.name}")
        return

    robust_genes = set(pd.read_csv(robust_path)['Gene'].tolist())
    optimal_genes = set(pd.read_csv(feature_path)['Optimal_Genes'].tolist())

    survivors = X_train[y_train[target_col] == 0]
    nonsurvivors = X_train[y_train[target_col] == 1]

    # ---------------------------------------------------------
    # 2. RECALCULATE GLOBAL LANDSCAPE STATISTICS
    # ---------------------------------------------------------
    print("    -> Calculating global Mean_Diff and FDR for background plotting...")
    p_values = []
    
    for gene in X_train.columns:
        stat, pval = ttest_ind(survivors[gene], nonsurvivors[gene], equal_var=False)
        p_values.append(pval)

    p_values = np.nan_to_num(p_values, nan=1.0) 
    _, fdr_pvals, _, _ = multipletests(p_values, alpha=0.05, method='fdr_bh')
    
    mean_diffs = nonsurvivors.mean() - survivors.mean()

    plot_df = pd.DataFrame({
        'Gene': X_train.columns,
        'Mean_Diff': mean_diffs.values,
        'neg_log10_padj': -np.log10(fdr_pvals + 1e-300)
    })

    # ---------------------------------------------------------
    # 3. CATEGORIZE FILTERING STAGES
    # ---------------------------------------------------------
    print("    -> Mapping visual hierarchy...")
    
    marker_optimal = f'Optimal Subset (RFECV) (n={len(optimal_genes)})'
    marker_robust = f'Significant DEGs (n={len(robust_genes)})'
    marker_bg = 'Background Transcriptome'

    def determine_status(row):
        if row['Gene'] in optimal_genes:
            return marker_optimal
        elif row['Gene'] in robust_genes:
            return marker_robust
        else:
            return marker_bg

    plot_df['Status'] = plot_df.apply(determine_status, axis=1)

    # ---------------------------------------------------------
    # 4. RENDER GEOMETRY 
    # ---------------------------------------------------------
    print("    -> Rendering layered Volcano plot...")
    sns.set_theme(style="ticks")
    
    fig, ax = plt.subplots(figsize=(8, 7))

    palette = {
        marker_bg: '#1a1a1a',      # Semi-transparent Black
        marker_robust: '#d62828',  # Crimson Red
        marker_optimal: '#2ecc71'  # Emerald Green
    }

    # Plot Background (Z-order 1)
    sns.scatterplot(
        data=plot_df[plot_df['Status'] == marker_bg],
        x='Mean_Diff', y='neg_log10_padj', color=palette[marker_bg],
        alpha=0.15, s=20, edgecolor='none', ax=ax, zorder=1, label=marker_bg
    )

    # Plot Significant DEGs (Z-order 3)
    sns.scatterplot(
        data=plot_df[plot_df['Status'] == marker_robust],
        x='Mean_Diff', y='neg_log10_padj', color=palette[marker_robust],
        alpha=0.6, s=45, edgecolor='white', linewidth=0.6, ax=ax, zorder=3, label=marker_robust
    )

    # Plot Optimal Subset (Z-order 5)
    sns.scatterplot(
        data=plot_df[plot_df['Status'] == marker_optimal],
        x='Mean_Diff', y='neg_log10_padj', color=palette[marker_optimal],
        alpha=0.85, s=110, edgecolor='white', linewidth=1.2, ax=ax, zorder=5, label=marker_optimal
    )

    # Mathematical threshold line
    ax.axhline(-np.log10(0.05), color='#777777', linestyle='--', linewidth=1, zorder=0)

    # ---------------------------------------------------------
    # 5. AESTHETICS & EXPORT
    # ---------------------------------------------------------
    ax.set_xlabel('Expression Shift (Z-Score Difference)', fontsize=12)
    ax.set_ylabel('-Log$_{10}$ (FDR Adjusted P-Value)', fontsize=12)
    
    ax.legend(
        title='Dimensionality Reduction Pipeline', 
        frameon=True, 
        facecolor='white',
        edgecolor='#dddddd',
        framealpha=0.95,
        loc='upper left', 
        fontsize=10, 
        title_fontsize=11
    )
    
    sns.despine()
    plt.tight_layout()

    out_path = FIG_OUT / "Fig3B.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[*] SUCCESS! Panel B saved to: {out_path.name}")

if __name__ == "__main__":
    main()