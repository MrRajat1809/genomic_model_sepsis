"""
07_distribution_analysis.py

The Final Methodology Step: Distribution of Variables.
Performs the Two-Sample Kolmogorov-Smirnov (KS) Test on the 26 Optimal
genes to prove their probability distributions are fundamentally distinct 
between Survivors and Non-Survivors.
Generates a wide, publication-ready 4x7 grid of unfilled KDE plots.
"""

import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ks_2samp

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
    print("[*] INITIATING VARIABLE DISTRIBUTION ANALYSIS (KS TEST)...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    print("    -> Loading Tensors and 26-Gene SHAP Consensus Rankings...")
    X_elite = pd.read_csv(DEG_DIR / "X_deg_master.csv.gz", compression='gzip')
    y = pd.read_csv(DATA_DIR / "y_master.csv")
    meta = pd.read_csv(DATA_DIR / "meta_master.csv")
    
    # Load all 26 optimal genes
    shap_df = pd.read_csv(DEG_DIR / "SHAP_Consensus_Feature_Importance_26Genes.csv")
    optimal_genes = shap_df['Gene'].tolist()

    # 2. Isolate Mega-Train Cohort
    train_mask = meta['Dataset'] != HOLDOUT_COHORT
    X_train = X_elite[train_mask]
    y_train = y[train_mask]['Mortality']

    survivors = X_train[y_train == 0]
    nonsurvivors = X_train[y_train == 1]

    # 3. Setup the Wide Plot Grid (4 rows x 7 columns = 28 subplots)
    print(f"    -> Running KS-Tests and plotting KDE distributions for all {len(optimal_genes)} Genes...")
    
    # Match the wide aspect ratio of the reference image
    fig, axes = plt.subplots(4, 7, figsize=(28, 14))
    axes = axes.flatten()
    
    # Clean white background, no grid lines
    sns.set_theme(style="white", rc={"axes.edgecolor": "black", "axes.linewidth": 1})

    ks_results = []

    for i, gene in enumerate(optimal_genes):
        s_data = survivors[gene]
        ns_data = nonsurvivors[gene]
        
        # Calculate Two-Sample Kolmogorov-Smirnov Test
        ks_stat, p_value = ks_2samp(s_data, ns_data)
        
        ks_results.append({
            'Gene': gene,
            'KS_Statistic': ks_stat,
            'P_Value': p_value
        })

        # Plot KDE (Unfilled lines matching the reference)
        sns.kdeplot(s_data, fill=False, color='tab:blue', label='Survivor', ax=axes[i], linewidth=2.5)
        sns.kdeplot(ns_data, fill=False, color='tab:orange', label='Non-Survivor', ax=axes[i], linewidth=2.5)
        
        # Match the reference title format exactly (no bold)
        axes[i].set_title(f"{gene} [KS stat: {ks_stat:.3f} pvalue: {p_value:.3e}]", fontsize=10)
        
        # Strip individual axis labels
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        
        # Add legend to every plot, upper right, small font
        axes[i].legend(loc='upper right', frameon=True, fontsize=8)

    # 4. Remove empty subplots (Since 26 genes < 28 grid slots)
    for j in range(len(optimal_genes), len(axes)):
        fig.delaxes(axes[j])

    # 5. Global Axis Labels (Bold removed)
    fig.supylabel('Probability Density Function', fontsize=22, x=0.01)
    fig.supxlabel('Z-Scored Gene Expression Values', fontsize=22, y=0.02)

    # Clean up layout
    plt.tight_layout(rect=[0.02, 0.04, 1, 1])
    
    # Save the Figure
    plot_path = FIG_OUT / "Fig8_KDE_Distributions_26Genes.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Save the Statistics
    ks_df = pd.DataFrame(ks_results)
    ks_df.to_csv(DEG_DIR / "KS_Test_Distributions_26Genes.csv", index=False)

    print(f"[*] SUCCESS! Distribution figures saved to {plot_path.name}")
    print("[*] THE METHODOLOGY IS COMPLETE.")

if __name__ == "__main__":
    main()