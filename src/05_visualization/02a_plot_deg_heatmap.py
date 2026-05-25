"""
02a_plot_deg_heatmap.py

Generates Panel A: Supervised and Directionally Sorted Heatmap.
Visualizes the expression landscape of the optimal 36 biomarker DEGs
using a custom transcriptomic colormap, sorted by regulation direction
and clinical outcome.
"""

import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from scipy.stats import zscore

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
    print("[*] Generating Figure 3 Panel A: Directionally Sorted Heatmap...")
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
    y_train = pd.read_csv(DATA_DIR / "y_train.csv")
    
    target_col = 'Mortality' if 'Mortality' in y_train.columns else y_train.columns[0]
    
    # ---------------------------------------------------------
    # 2. CALCULATE REGULATION DIRECTION & SORT
    # ---------------------------------------------------------
    print("    -> Calculating group means to determine regulation direction...")
    
    surv_means = X_train[y_train[target_col] == 0][optimal_genes].mean()
    nsurv_means = X_train[y_train[target_col] == 1][optimal_genes].mean()
    
    expression_diff = nsurv_means - surv_means
    sorted_genes = expression_diff.sort_values(ascending=True).index.tolist()

    # ---------------------------------------------------------
    # 3. DATA PREPARATION & Z-SCORING
    # ---------------------------------------------------------
    print("    -> Structuring matrix and applying Z-score normalization...")
    
    X_opt = X_train[sorted_genes].copy()
    X_zscored = X_opt.apply(zscore)
    X_zscored['Mortality'] = y_train[target_col].values
    
    X_sorted = X_zscored.sort_values(by='Mortality')
    
    plot_matrix = X_sorted.drop(columns=['Mortality'])[sorted_genes].T 
    outcome_array = X_sorted['Mortality'].values.reshape(1, -1)

    # ---------------------------------------------------------
    # 4. CUSTOM TRANSCRIPTOMIC COLORMAP
    # ---------------------------------------------------------
    gbr_cmap = LinearSegmentedColormap.from_list(
        "GreenBlackRed", ["#00FF00", "#000000", "#FF0000"]
    )

    # ---------------------------------------------------------
    # 5. GRIDSPEC LAYOUT
    # ---------------------------------------------------------
    print("    -> Rendering GridSpec layout...")
    
    sns.set_theme(style="white")
    fig = plt.figure(figsize=(6.5, 9.0))
    
    gs = fig.add_gridspec(
        nrows=3, ncols=1, 
        height_ratios=[0.02, 1.0, 0.025], 
        hspace=0.12 
    )
    
    ax_annot = fig.add_subplot(gs[0])
    ax_heat = fig.add_subplot(gs[1])
    ax_cbar = fig.add_subplot(gs[2])

    # ---------------------------------------------------------
    # 6. RENDER ELEMENTS
    # ---------------------------------------------------------
    outcome_cmap = ListedColormap(['#4a6fe3', '#db4325'])
    sns.heatmap(
        outcome_array, ax=ax_annot, cmap=outcome_cmap, 
        cbar=False, xticklabels=False, yticklabels=False
    )
    
    sns.heatmap(
        plot_matrix, ax=ax_heat, cmap=gbr_cmap, center=0,
        vmin=-2.0, vmax=2.0, cbar_ax=ax_cbar,
        cbar_kws={"orientation": "horizontal"},
        xticklabels=False, yticklabels=True
    )

    split_idx = (X_sorted['Mortality'] == 0).sum()
    zero_cross_idx = (expression_diff.sort_values(ascending=True) < 0).sum()

    ax_annot.axvline(split_idx, color='white', linewidth=1.5) 
    ax_heat.axvline(split_idx, color='white', linewidth=1.5)
    ax_heat.axhline(zero_cross_idx, color='#888888', linewidth=1.2, linestyle='--')

    ax_heat.set_yticklabels(ax_heat.get_ymajorticklabels(), fontsize=8, rotation=0)
    ax_heat.set_xlabel("Patients (Grouped by Clinical Outcome)", fontsize=10, labelpad=8)
    ax_heat.set_ylabel("") 
    ax_heat.tick_params(axis='both', length=0)
    
    ax_cbar.set_xlabel("Standardized Expression (Z-Score)", fontsize=9, labelpad=6)
    ax_cbar.tick_params(labelsize=8, length=3)

    # ---------------------------------------------------------
    # 7. MARGIN ANNOTATIONS & LEGEND
    # ---------------------------------------------------------
    num_down = zero_cross_idx
    num_up = len(optimal_genes) - num_down
    MARGIN_TEXT_X_OFFSET = -0.15
    
    ax_heat.text(
        MARGIN_TEXT_X_OFFSET, num_down / 2.0, f"Downregulated (n={num_down})",
        transform=ax_heat.get_yaxis_transform(),
        rotation=90, ha='center', va='center', 
        fontsize=10, color='#009900', style='italic'
    )
    
    ax_heat.text(
        MARGIN_TEXT_X_OFFSET, num_down + (num_up / 2.0), f"Upregulated (n={num_up})",
        transform=ax_heat.get_yaxis_transform(),
        rotation=90, ha='center', va='center', 
        fontsize=10, color='#cc0000', style='italic'
    )

    surv_patch = mpatches.Patch(color='#4a6fe3', label='Survivor')
    nonsurvivor_patch = mpatches.Patch(color='#db4325', label='Non-Survivor')
    
    ax_annot.legend(
        handles=[surv_patch, nonsurvivor_patch],
        loc="lower right", bbox_to_anchor=(1.0, 1.2),
        ncol=2, frameon=False, fontsize=9
    )

    # ---------------------------------------------------------
    # 8. EXPORT
    # ---------------------------------------------------------
    out_path = FIG_OUT / "Fig3A.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[*] SUCCESS! Panel A saved to: {out_path.name}")

if __name__ == "__main__":
    main()