"""
02_distribution_analysis.py

Final Feature Distribution Validation.
Performs the Two-Sample Kolmogorov-Smirnov (KS) Test on the optimal biomarker 
signature to statistically validate that the continuous probability distributions 
of these features are fundamentally distinct between Survivors and Non-Survivors.
Generates an automated, dynamically scaled grid of KDE distributions for publication.
"""

import math
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import ks_2samp

# Suppress warnings for clean execution logging
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
# Resolves to Multimodal_prediction_sepsis/
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"
DEG_DIR = BASE_DIR / "data" / "processed" / "deg_tensors"
FEATURE_DIR = BASE_DIR / "outputs" / "features"
FIG_OUT = BASE_DIR / "outputs" / "figures"

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    """Main execution workflow."""
    print("[*] Initiating statistical distribution analysis (KS Test)...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD DATA & FEATURES
    # ---------------------------------------------------------
    print("    -> Loading isolated training tensors and optimal features...")
    
    # Load optimal feature list
    feature_path = FEATURE_DIR / "optimal_biomarker_panel.csv"
    if not feature_path.exists():
        raise FileNotFoundError(f"[ERROR] Missing optimal feature list: {feature_path.name}")
    
    optimal_genes = pd.read_csv(feature_path)['Optimal_Genes'].tolist()
    num_genes = len(optimal_genes)

    # Load isolated training data
    X_train_full = pd.read_csv(DEG_DIR / "X_train_deg_subset.csv.gz", compression='gzip')
    y_train_df = pd.read_csv(DATA_DIR / "y_train.csv")
    
    target_col = 'Mortality' if 'Mortality' in y_train_df.columns else y_train_df.columns[0]
    y_train = y_train_df[target_col].astype(int).values

    # Isolate outcome cohorts
    survivors = X_train_full[y_train == 0]
    nonsurvivors = X_train_full[y_train == 1]
    
    print(f"       - Evaluating {num_genes} genes across {len(y_train)} patients.")

    # ---------------------------------------------------------
    # 2. DYNAMIC GRID SETUP
    # ---------------------------------------------------------
    print("    -> Computing Two-Sample KS Tests and generating KDE distributions...")
    
    # Calculate optimal grid dimensions dynamically
    cols = 6
    rows = math.ceil(num_genes / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.5))
    axes = axes.flatten()
    
    sns.set_theme(style="white", rc={"axes.edgecolor": "#333333", "axes.linewidth": 1.2})

    ks_results = []

    # ---------------------------------------------------------
    # 3. STATISTICAL TESTING & PLOTTING
    # ---------------------------------------------------------
    for i, gene in enumerate(optimal_genes):
        s_data = survivors[gene]
        ns_data = nonsurvivors[gene]
        
        # Two-Sample Kolmogorov-Smirnov Test
        ks_stat, p_value = ks_2samp(s_data, ns_data)
        
        ks_results.append({
            'Gene': gene,
            'KS_Statistic': ks_stat,
            'P_Value': p_value
        })

        # Plot Probability Density (Unfilled)
        sns.kdeplot(s_data, fill=False, color='#4a6fe3', label='Survivor', ax=axes[i], linewidth=2.5)
        sns.kdeplot(ns_data, fill=False, color='#db4325', label='Non-Survivor', ax=axes[i], linewidth=2.5)
        
        # Format Title with Statistics
        axes[i].set_title(f"{gene} [KS: {ks_stat:.3f} | p: {p_value:.2e}]", fontsize=11, color='#333333')
        
        axes[i].set_xlabel('')
        axes[i].set_ylabel('')
        
        # Clean legend integration
        axes[i].legend(loc='upper right', frameon=True, fontsize=8, edgecolor='#cccccc')

    # ---------------------------------------------------------
    # 4. AESTHETIC CLEANUP & EXPORT
    # ---------------------------------------------------------
    # Delete unused axes if num_genes is not a perfect multiple of cols
    for j in range(num_genes, len(axes)):
        fig.delaxes(axes[j])

    # Global Axis Labels
    fig.supylabel('Probability Density Function', fontsize=18, x=0.01, fontweight='bold', color='#222222')
    fig.supxlabel('Gene Expression (Z-Scored)', fontsize=18, y=0.01, fontweight='bold', color='#222222')

    plt.tight_layout(rect=[0.02, 0.03, 1, 1])
    
    # Export Plot
    plot_path = FIG_OUT / f"Fig_KDE_Distributions_{num_genes}Genes.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Export Statistics
    ks_df = pd.DataFrame(ks_results).sort_values(by='KS_Statistic', ascending=False)
    csv_path = FEATURE_DIR / f"ks_test_statistics_{num_genes}genes.csv"
    ks_df.to_csv(csv_path, index=False)

    print(f"       [+] Density distributions saved to: {plot_path.name}")
    print(f"       [+] Statistical registry saved to : {csv_path.name}")
    
    print("\n" + "=" * 65)
    print("[*] METHODOLOGICAL PIPELINE COMPLETE.")
    print("=" * 65)

# ==========================================
# EXECUTION GUARD
# ==========================================
if __name__ == "__main__":
    main()