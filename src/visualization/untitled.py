"""
01_batch_effect_diagnostics.py

Generates the Gold Standard PCA evaluation to prove successful batch effect
removal across the 7 independent multi-center cohorts.
Calculates the Silhouette Score to mathematically quantify dataset mixing.
"""

import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"
FIG_OUT = BASE_DIR / "outputs" / "figures"

def main():
    print("[*] INITIATING BATCH EFFECT DIAGNOSTICS (PCA)...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # 1. Load the fully processed Master Tensors
    print("    -> Loading harmonized multi-center tensors...")
    X = pd.read_csv(DATA_DIR / "X_master.csv.gz", compression='gzip')
    y = pd.read_csv(DATA_DIR / "y_master.csv")
    meta = pd.read_csv(DATA_DIR / "meta_master.csv")

    # 2. Run Principal Component Analysis (PCA)
    print("    -> Calculating Principal Components...")
    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(X)
    
    # Create a dataframe for plotting
    pca_df = pd.DataFrame(data=pca_results, columns=['PC1', 'PC2'])
    pca_df['Cohort'] = meta['Dataset']
    pca_df['Mortality'] = y['Mortality'].map({0: 'Survivor', 1: 'Non-Survivor'})
    
    # Calculate Variance Explained
    var_explained = pca.explained_variance_ratio_ * 100

    # 3. Calculate the Silhouette Score for Batch Mixing
    print("    -> Calculating Mathematical Batch Mixing Score...")
    # We use a random subsample if the dataset is huge, but 1636 is small enough to compute fully
    sil_score = silhouette_score(pca_results, pca_df['Cohort'])
    print(f"       [SILHOUETTE SCORE]: {sil_score:.4f} (Closer to 0 is perfect mixing)")

    # 4. Generate the Visualization
    print("    -> Plotting aesthetic PCA grids...")
    
    # Set a clean, minimalist aesthetic
    sns.set_theme(style="ticks", rc={"axes.linewidth": 1.2, "axes.edgecolor": "#333333"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Plot A: Colored by Cohort (Looking for "The Smoothie Effect")
    sns.scatterplot(
        x='PC1', y='PC2', 
        hue='Cohort', 
        data=pca_df, 
        palette='Set2', 
        alpha=0.7, 
        s=40, 
        edgecolor='none',
        ax=ax1
    )
    
    ax1.set_title('Post-ComBat Harmonization: Mixing by Cohort', fontsize=14, pad=15)
    ax1.set_xlabel(f'Principal Component 1 ({var_explained[0]:.1f}%)', fontsize=12)
    ax1.set_ylabel(f'Principal Component 2 ({var_explained[1]:.1f}%)', fontsize=12)
    ax1.legend(title='Clinical Cohort', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    
    # Add the Silhouette Score annotation to the plot
    ax1.text(
        0.05, 0.95, f'Batch Silhouette Score: {sil_score:.3f}\n(Ideal ~ 0.0)', 
        transform=ax1.transAxes, 
        fontsize=11, 
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#cccccc', alpha=0.9)
    )

    # Plot B: Colored by Mortality (Looking for biological structure)
    sns.scatterplot(
        x='PC1', y='PC2', 
        hue='Mortality', 
        data=pca_df, 
        palette={'Survivor': '#4a6fe3', 'Non-Survivor': '#db4325'}, 
        alpha=0.7, 
        s=40, 
        edgecolor='none',
        ax=ax2
    )
    
    ax2.set_title('Biological Signal Preservation: Mortality Status', fontsize=14, pad=15)
    ax2.set_xlabel(f'Principal Component 1 ({var_explained[0]:.1f}%)', fontsize=12)
    ax2.set_ylabel(f'Principal Component 2 ({var_explained[1]:.1f}%)', fontsize=12)
    ax2.legend(title='Clinical Outcome', loc='upper right', frameon=True, edgecolor='#cccccc')

    sns.despine()
    plt.tight_layout()
    
    # Save the output
    plot_path = FIG_OUT / "Fig1_ComBat_PCA2.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[*] SUCCESS! Quality Control check complete. Saved to {plot_path.name}")

if __name__ == "__main__":
    main()