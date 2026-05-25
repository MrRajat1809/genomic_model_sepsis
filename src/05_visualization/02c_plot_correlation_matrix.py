"""
02c_plot_correlation_matrix.py

Generates Panel C: Co-expression Correlation Matrix.
Mathematically calculates hierarchical linkage to group genes into biological 
modules, but plots strictly the raw sorted matrix to eliminate dendrogram clutter.
Utilizes the RdBu_r palette and a spacious layout with a right-docked colorbar.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as hc
import scipy.spatial as sp
import seaborn as sns

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
DEG_DIR = BASE_DIR / "data" / "processed" / "deg_tensors"
FEATURE_DIR = BASE_DIR / "outputs" / "features"
FIG_OUT = BASE_DIR / "outputs" / "figures"

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("[*] Generating Figure 3 Panel C: Structured Correlation Matrix...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD OPTIMAL FEATURES & TENSORS
    # ---------------------------------------------------------
    feature_path = FEATURE_DIR / "optimal_biomarker_panel.csv"
    if not feature_path.exists():
        print(f"[ERROR] Missing optimal feature list: {feature_path.name}")
        return
        
    optimal_genes = pd.read_csv(feature_path)['Optimal_Genes'].tolist()

    print("    -> Loading harmonized training tensors...")
    X_train = pd.read_csv(DEG_DIR / "X_train_deg_subset.csv.gz", compression='gzip')

    # ---------------------------------------------------------
    # 2. MATRIX COMPUTATION & MATHEMATICAL CLUSTERING
    # ---------------------------------------------------------
    print("    -> Computing pairwise Pearson correlation and biological linkage...")
    
    X_opt = X_train[optimal_genes].copy()
    corr_matrix = X_opt.corr(method='pearson')
    
    # Convert correlation to a distance metric (1 = highly correlated, 0 = no correlation)
    dist_matrix = np.clip(1.0 - corr_matrix.values, 0, 2)
    
    # Condense the distance matrix and compute hierarchical linkage (Ward's method)
    condensed_dist = sp.distance.squareform(dist_matrix, checks=False)
    linkage_matrix = hc.linkage(condensed_dist, method='ward')
    
    # Extract the sorted gene order from the theoretical dendrogram
    order = hc.leaves_list(linkage_matrix)
    sorted_genes = corr_matrix.columns[order]
    
    # Rebuild the final matrix perfectly sorted into biological modules
    sorted_corr = corr_matrix.loc[sorted_genes, sorted_genes]

    # ---------------------------------------------------------
    # 3. GRIDSPEC LAYOUT (Spacious & Clean)
    # ---------------------------------------------------------
    print("    -> Rendering GridSpec layout with right-docked colorbar...")
    sns.set_theme(style="white")
    
    fig = plt.figure(figsize=(8.5, 8.0))
    
    gs = fig.add_gridspec(
        nrows=1, ncols=2, 
        width_ratios=[1.0, 0.03], 
        wspace=0.04               
    )
    
    ax_heat = fig.add_subplot(gs[0])
    ax_cbar = fig.add_subplot(gs[1])

    # ---------------------------------------------------------
    # 4. RENDER
    # ---------------------------------------------------------
    sns.heatmap(
        sorted_corr, 
        cmap="RdBu_r", 
        center=0, 
        vmin=-1.0, vmax=1.0,
        cbar_ax=ax_cbar,
        ax=ax_heat,
        linewidths=0.2,            
        linecolor='#eeeeee'
    )

    # ---------------------------------------------------------
    # 5. AESTHETICS & EXPORT
    # ---------------------------------------------------------
    ax_heat.set_xticklabels(ax_heat.get_xmajorticklabels(), fontsize=9, rotation=90)
    ax_heat.set_yticklabels(ax_heat.get_ymajorticklabels(), fontsize=9, rotation=0)
    
    ax_heat.set_xlabel("")
    ax_heat.set_ylabel("")

    ax_cbar.set_ylabel("Pearson Correlation (r)", fontsize=11, labelpad=12, rotation=270, va='bottom')
    ax_cbar.tick_params(labelsize=9, length=3)

    for spine in ax_heat.spines.values():
        spine.set_visible(True)
        spine.set_color('#cccccc')
        spine.set_linewidth(1.0)

    out_path = FIG_OUT / "Fig3C.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[*] SUCCESS! Panel C saved to: {out_path.name}")

if __name__ == "__main__":
    main()