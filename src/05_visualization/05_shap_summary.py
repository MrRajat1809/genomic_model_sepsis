"""
05_shap_summary.py

Generates Figure 6: Consensus SHAP Summary Plot.
Loads the pre-computed raw SHAP matrices to generate a globally stable,
clinically interpretable visualization of the non-linear decision thresholds.
Visually optimized with a compact canvas, translucent "bubble" aesthetics, 
and uniform, non-bold typography for a strict journal-ready appearance.
"""

import warnings
from pathlib import Path

import matplotlib.colors as mcolors
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

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
    print("[*] Generating Figure 6: Consensus SHAP Summary Plot...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD DATA & PRE-COMPUTED SHAP MATRICES
    # ---------------------------------------------------------
    feature_path = FEATURE_DIR / "optimal_biomarker_panel.csv"
    shap_matrix_path = FEATURE_DIR / "raw_consensus_shap_values.npy"
    
    if not feature_path.exists() or not shap_matrix_path.exists():
        print("[ERROR] Missing required SHAP data. Run calculation script first.")
        return

    optimal_genes = pd.read_csv(feature_path)['Optimal_Genes'].tolist()
    num_genes = len(optimal_genes)

    print("    -> Loading training tensors and SHAP consensus arrays...")
    X_train_full = pd.read_csv(DEG_DIR / "X_train_deg_subset.csv.gz", compression='gzip')
    X_train = X_train_full[optimal_genes]
    
    consensus_shap_values = np.load(shap_matrix_path)

    # ---------------------------------------------------------
    # 2. GENERATE AESTHETIC SUMMARY PLOT
    # ---------------------------------------------------------
    print("    -> Rendering clinical interpretability visualization...")
    
    # Transitioning smoothly from Survivor Blue -> Deep Violet -> Non-Survivor Red
    aesthetic_colors = ["#4a6fe3", "#8a71b3", "#db4325"]
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("AestheticCmap", aesthetic_colors)

    # Clamped the plot_size to a compact 7.5 x 6.5 inches
    shap.summary_plot(
        consensus_shap_values, 
        X_train, 
        max_display=num_genes,  
        show=False,
        plot_size=(7.5, 6.5), 
        cmap=custom_cmap
    )
    
    ax = plt.gca()

    # Iterate through the plot collections to force the "bubble" aesthetic
    for collection in ax.collections:
        if isinstance(collection, mcoll.PathCollection):
            collection.set_edgecolor('white')
            collection.set_linewidth(0.4)
            collection.set_alpha(0.8)
    
    # ---------------------------------------------------------
    # 3. FORMATTING, LABELS & EXPORT
    # ---------------------------------------------------------
    # Clean up the bounding box
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    
    # Typography
    ax.tick_params(axis='both', labelsize=9, width=1.0, length=4, colors='#222222')
    ax.set_xlabel('SHAP Value (Impact on Model Output)', fontsize=11, labelpad=10, color='#111111')
    
    # Inject a prominent, unmistakable central baseline
    ax.axvline(x=0, color='#111111', linestyle='--', linewidth=1.5, zorder=0)

    # Clinical interpretation labels
    ax.text(
        0.0, 1.02, "← Favors Survival",
        transform=ax.transAxes,
        fontsize=9, color='#4a6fe3', ha='left', va='bottom'
    )
    
    ax.text(
        1.0, 1.02, "Favors Mortality →",
        transform=ax.transAxes,
        fontsize=9, color='#db4325', ha='right', va='bottom'
    )

    # Safely format the SHAP colorbar on the right
    cb = plt.gcf().axes[-1] 
    cb.tick_params(labelsize=9, colors='#222222', width=0)
    cb.set_ylabel('Gene Expression (Z-Score)', fontsize=11, labelpad=12, color='#111111')
    
    # Cleanly remove the colorbar border using spines
    for spine in cb.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    
    out_path = FIG_OUT / "Fig6_SHAP_Summary.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[*] SUCCESS! Figure 6 saved to: {out_path.name}")

if __name__ == "__main__":
    main()