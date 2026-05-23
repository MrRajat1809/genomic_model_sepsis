"""
02_plot_pca_biology_preservation.py

Generates post-harmonization quality control visualizations.
Calculates the Silhouette Score to mathematically quantify dataset mixing, 
and generates a dual-panel PCA figure to prove that technical batches are 
integrated while the biological signal (Mortality) remains structurally preserved.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Suppress warnings for clean execution logging
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
# Resolves to Multimodal_prediction_sepsis/
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"
FIG_OUT = BASE_DIR / "outputs" / "figures"

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    """Main execution workflow."""
    print("[*] Initiating post-harmonization quality control diagnostics...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD HARMONIZED TENSORS
    # ---------------------------------------------------------
    print("    -> Loading harmonized multi-center tensors...")
    X = pd.read_csv(DATA_DIR / "X_atlas.csv.gz", compression='gzip')
    y = pd.read_csv(DATA_DIR / "y_atlas.csv")
    meta = pd.read_csv(DATA_DIR / "meta_atlas.csv")

    # ---------------------------------------------------------
    # 2. COMPUTE PRINCIPAL COMPONENTS
    # ---------------------------------------------------------
    print("    -> Calculating Principal Components...")
    pca = PCA(n_components=2, random_state=42)
    pca_results = pca.fit_transform(X)
    
    pca_df = pd.DataFrame(data=pca_results, columns=['PC1', 'PC2'])
    pca_df['Cohort'] = meta['Dataset']
    pca_df['Mortality'] = y['Mortality'].map({0: 'Survivor', 1: 'Non-Survivor'})
    
    var_explained = pca.explained_variance_ratio_ * 100

    # ---------------------------------------------------------
    # 3. MATHEMATICAL BATCH MIXING QUANTIFICATION
    # ---------------------------------------------------------
    print("    -> Computing Silhouette Score for cohort mixing evaluation...")
    sil_score = silhouette_score(pca_results, pca_df['Cohort'])
    print(f"       [METRIC] Global Silhouette Score: {sil_score:.4f} (Optimal near 0.0)")

    # ---------------------------------------------------------
    # 4. GENERATE PUBLICATION FIGURE
    # ---------------------------------------------------------
    print("    -> Generating aesthetic PCA grids...")
    
    sns.set_theme(style="ticks", rc={"axes.linewidth": 1.2, "axes.edgecolor": "#333333"})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

    # Panel A: Cohort Mixing & Silhouette Annotation
    sns.scatterplot(
        x='PC1', y='PC2', hue='Cohort', data=pca_df, 
        palette='Set2', alpha=0.7, s=40, edgecolor='none', ax=ax1
    )
    
    ax1.set_title('A. Post-Harmonization: Technical Mixing by Cohort', fontsize=14, pad=15)
    ax1.set_xlabel(f'Principal Component 1 ({var_explained[0]:.1f}%)', fontsize=12)
    ax1.set_ylabel(f'Principal Component 2 ({var_explained[1]:.1f}%)', fontsize=12)
    ax1.legend(title='Clinical Cohort', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    
    # Annotate Silhouette Score
    ax1.text(
        0.05, 0.95, f'Batch Silhouette Score: {sil_score:.3f}\n(Ideal ~ 0.0)', 
        transform=ax1.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='#cccccc', alpha=0.9)
    )

    # Panel B: Biological Signal Preservation
    sns.scatterplot(
        x='PC1', y='PC2', hue='Mortality', data=pca_df, 
        palette={'Survivor': '#4a6fe3', 'Non-Survivor': '#db4325'}, 
        alpha=0.7, s=40, edgecolor='none', ax=ax2
    )
    
    ax2.set_title('B. Biological Signal Preservation: Mortality Status', fontsize=14, pad=15)
    ax2.set_xlabel(f'Principal Component 1 ({var_explained[0]:.1f}%)', fontsize=12)
    ax2.set_ylabel(f'Principal Component 2 ({var_explained[1]:.1f}%)', fontsize=12)
    ax2.legend(title='Clinical Outcome', loc='upper right', frameon=True, edgecolor='#cccccc')

    sns.despine()
    plt.tight_layout()
    
    plot_path = FIG_OUT / "Fig_PCA_Mixing_vs_Mortality.pdf"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[*] Diagnostics complete. Figure saved to: {plot_path.name}")

# ==========================================
# EXECUTION GUARD
# ==========================================
if __name__ == "__main__":
    main()