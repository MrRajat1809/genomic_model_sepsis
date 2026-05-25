"""
04_kde_distributions.py

Generates Figure 5: High-Density KDE Distribution Grid.
Reads the previously calculated KS Test statistics to annotate the plots,
validating the structural divergence of biomarker signals. Sorts the grid 
by descending KS score to establish a visual hierarchy of signal strength.
"""

import math
import warnings
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

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
    print("[*] Generating Figure 5: KDE Distribution Grids...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD FEATURES & STATISTICS
    # ---------------------------------------------------------
    feature_path = FEATURE_DIR / "optimal_biomarker_panel.csv"
    if not feature_path.exists():
        print(f"[ERROR] Missing optimal feature list: {feature_path.name}")
        return
        
    optimal_genes = pd.read_csv(feature_path)['Optimal_Genes'].tolist()
    num_genes = len(optimal_genes)

    stats_path = FEATURE_DIR / f"ks_test_statistics_{num_genes}genes.csv"
    if not stats_path.exists():
        print(f"[ERROR] Missing KS statistics: {stats_path.name}. Run 02a script first.")
        return

    ks_df = pd.read_csv(stats_path).set_index('Gene')
    optimal_genes = sorted(optimal_genes, key=lambda g: ks_df.loc[g, 'KS_Statistic'], reverse=True)

    # ---------------------------------------------------------
    # 2. LOAD TRAINING TENSORS
    # ---------------------------------------------------------
    print("    -> Loading isolated training tensors...")
    X_train_full = pd.read_csv(DEG_DIR / "X_train_deg_subset.csv.gz", compression='gzip')
    y_train_df = pd.read_csv(DATA_DIR / "y_train.csv")

    target_col = 'Mortality' if 'Mortality' in y_train_df.columns else y_train_df.columns[0]
    y_train = y_train_df[target_col].astype(int).values

    survivors = X_train_full[y_train == 0]
    nonsurvivors = X_train_full[y_train == 1]

    # ---------------------------------------------------------
    # 3. HIGH-DENSITY 6x6 GRID SETUP
    # ---------------------------------------------------------
    print(f"    -> Rendering optimized 6x6 KDE trellis for {num_genes} genes...")
    cols = 6
    rows = math.ceil(num_genes / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(8.0, 9.5))
    axes = axes.flatten()
    
    sns.set_theme(style="white", rc={"axes.edgecolor": "#dddddd", "axes.linewidth": 0.8})

    # ---------------------------------------------------------
    # 4. PLOTTING
    # ---------------------------------------------------------
    for i, gene in enumerate(optimal_genes):
        s_data = survivors[gene]
        ns_data = nonsurvivors[gene]
        
        ks_stat = ks_df.loc[gene, 'KS_Statistic']
        p_value = ks_df.loc[gene, 'P_Value']

        sns.kdeplot(s_data, fill=True, alpha=0.1, color='#4a6fe3', ax=axes[i], linewidth=0.8)
        sns.kdeplot(ns_data, fill=True, alpha=0.1, color='#db4325', ax=axes[i], linewidth=0.8)
        
        axes[i].set_title(gene, fontsize=8, pad=3)
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        
        if axes[i].legend_:
            axes[i].legend_.remove()
            
        stat_text = f"KS: {ks_stat:.2f}\np: {p_value:.1e}"
        axes[i].text(0.95, 0.90, stat_text, transform=axes[i].transAxes, 
                     fontsize=5.5, ha='right', va='top', 
                     bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
        
        axes[i].tick_params(axis='both', which='major', labelsize=5)

    for j in range(num_genes, len(axes)):
        fig.delaxes(axes[j])

    # ---------------------------------------------------------
    # 5. GLOBAL FORMATTING & LEGEND
    # ---------------------------------------------------------
    fig.supylabel('Probability Density Function', fontsize=10, x=0.02, color='#222222')
    fig.supxlabel('Gene Expression (Z-Scored)', fontsize=10, y=0.02, color='#222222')

    surv_patch = mpatches.Patch(color='#4a6fe3', alpha=0.6, label='Survivor')
    nsurv_patch = mpatches.Patch(color='#db4325', alpha=0.6, label='Non-Survivor')
    
    fig.legend(handles=[surv_patch, nsurv_patch], loc='upper center', bbox_to_anchor=(0.5, 0.98), 
               ncol=2, fontsize=8, frameon=False)

    plt.subplots_adjust(top=0.92, bottom=0.06, left=0.08, right=0.98, hspace=0.6, wspace=0.4)
    
    # ---------------------------------------------------------
    # 6. EXPORT
    # ---------------------------------------------------------
    plot_path = FIG_OUT / "Fig5_KS_Distribution.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[*] SUCCESS! Density distributions saved to: {plot_path.name}")

if __name__ == "__main__":
    main()