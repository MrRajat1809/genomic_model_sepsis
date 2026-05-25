"""
01b_pca_biology_preservation.py

Generates Panel B: Post-Harmonization Quality Control.
Calculates the Silhouette Score to mathematically quantify dataset mixing, 
and generates a dual-panel PCA figure to prove that technical batches are 
integrated while the biological signal (Mortality) remains preserved.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
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

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("[*] Generating Figure 2 Panel B: Quality Control Diagnostics...")
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

    # ---------------------------------------------------------
    # 3. MATHEMATICAL BATCH MIXING QUANTIFICATION
    # ---------------------------------------------------------
    print("    -> Computing Silhouette Score for cohort mixing...")
    sil_score = silhouette_score(pca_results, pca_df['Cohort'])
    print(f"       [METRIC] Global Silhouette Score: {sil_score:.4f} (Optimal near 0.0)")

    # ---------------------------------------------------------
    # 4. RENDER VISUALIZATION
    # ---------------------------------------------------------
    print("    -> Rendering PCA grids...")
    sns.set_theme(style="white")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Synchronize palette with Fig2A to ensure cross-panel consistency
    cohort_palette = sns.color_palette("husl", len(pca_df['Cohort'].unique()))

    # Panel C (Left): Cohort Mixing & Silhouette Annotation
    sns.scatterplot(
        x='PC1', y='PC2', hue='Cohort', data=pca_df, 
        palette=cohort_palette, alpha=0.7, s=50, ax=axes[0]
    )
    
    axes[0].set_title('C. Technical Mixing by Cohort', fontsize=14)
    axes[0].set_xlabel('Principal Component 1', fontsize=12)
    axes[0].set_ylabel('Principal Component 2', fontsize=12)
    axes[0].legend(title='Clinical Cohort', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    sns.despine(ax=axes[0])
    
    # Annotate Silhouette Score with a clean, unobtrusive bounding box
    axes[0].text(
        0.05, 0.95, f'Batch Silhouette Score: {sil_score:.3f}\n(Ideal ~ 0.0)', 
        transform=axes[0].transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='square,pad=0.6', facecolor='#f9f9f9', edgecolor='#dddddd', alpha=0.9)
    )

    # Panel D (Right): Biological Signal Preservation
    sns.scatterplot(
        x='PC1', y='PC2', hue='Mortality', data=pca_df, 
        palette={'Survivor': '#4a6fe3', 'Non-Survivor': '#db4325'}, 
        alpha=0.7, s=50, ax=axes[1]
    )
    
    axes[1].set_title('D. Biological Signal Preservation (Mortality)', fontsize=14)
    axes[1].set_xlabel('Principal Component 1', fontsize=12)
    axes[1].set_ylabel('Principal Component 2', fontsize=12)
    axes[1].legend(title='Clinical Outcome', loc='upper right', frameon=False)
    sns.despine(ax=axes[1])

    plt.tight_layout()
    
    # ---------------------------------------------------------
    # 5. EXPORT
    # ---------------------------------------------------------
    out_path = FIG_OUT / "Fig2B.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[*] SUCCESS! Panel B saved to: {out_path.name}")

if __name__ == "__main__":
    main()