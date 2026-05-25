"""
01a_pca_batch_effect.py

Generates Panel A: Comparative PCA Visualization.
Contrasts the raw, uncorrected expression tensor against the final 
Empirical Bayes (ComBat) corrected tensor to visually demonstrate 
the neutralization of technical batch effects across independent cohorts.
"""

import gc
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
MATRIX_DIR = BASE_DIR / "data" / "processed" / "mapped_matrices"
ML_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"
FIG_OUT = BASE_DIR / "outputs" / "figures"

# List of any files to exclude from the raw matrix compilation
BLACKLIST = []

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("[*] Generating Figure 2 Panel A: Comparative PCA Diagnostics...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD COMBAT TENSOR ("AFTER")
    # ---------------------------------------------------------
    print("    -> Loading ComBat-corrected integrated tensor...")
    X_combat = pd.read_csv(ML_DIR / "X_atlas.csv.gz", compression='gzip')
    meta = pd.read_csv(ML_DIR / "meta_atlas.csv")
    common_genes = list(X_combat.columns)

    # ---------------------------------------------------------
    # 2. REBUILD RAW TENSOR ("BEFORE")
    # ---------------------------------------------------------
    print("    -> Reconstructing raw Log1p expression tensor for baseline comparison...")
    raw_X = []
    matrix_files = sorted([f.name for f in MATRIX_DIR.iterdir() 
                           if f.name.endswith('_mapped.csv.gz') and f.name not in BLACKLIST])
    
    patient_to_cohort = dict(zip(meta['Patient_ID'], meta['Dataset']))

    for file_name in matrix_files:
        df = pd.read_csv(MATRIX_DIR / file_name, index_col=0, compression='gzip')
        df.index = df.index.astype(str).str.strip().str.upper()
        df = df[~df.index.duplicated(keep='first')]
        df = df.loc[common_genes].T
        
        # Apply Log1p transform to standardize scaling prior to PCA
        df_log = np.log1p(df.clip(lower=0)) 
        
        for patient_id in df.index:
            if patient_id in patient_to_cohort:
                raw_X.append(df_log.loc[patient_id].values)
                
        del df, df_log
        gc.collect()

    df_meta = pd.DataFrame({'Cohort': meta['Dataset']})

    # ---------------------------------------------------------
    # 3. COMPUTE PRINCIPAL COMPONENTS
    # ---------------------------------------------------------
    print("    -> Computing Principal Components for both tensors...")
    pca = PCA(n_components=2, random_state=42)
    
    pca_raw = pca.fit_transform(raw_X)
    pca_combat = pca.fit_transform(X_combat.values)

    df_meta['Raw_PC1'] = pca_raw[:, 0]
    df_meta['Raw_PC2'] = pca_raw[:, 1]
    df_meta['ComBat_PC1'] = pca_combat[:, 0]
    df_meta['ComBat_PC2'] = pca_combat[:, 1]

    # ---------------------------------------------------------
    # 4. RENDER VISUALIZATION
    # ---------------------------------------------------------
    print("    -> Generating comparative PCA visualization...")
    sns.set_theme(style="white")
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    palette = sns.color_palette("husl", len(df_meta['Cohort'].unique()))

    # Panel A: Uncorrected Baseline
    sns.scatterplot(
        data=df_meta, x='Raw_PC1', y='Raw_PC2', hue='Cohort', 
        alpha=0.7, s=50, ax=axes[0], palette=palette
    )
    axes[0].set_title('A. Baseline Variance (Raw Log1p Expression)', fontsize=14)
    axes[0].set_xlabel('Principal Component 1', fontsize=12)
    axes[0].set_ylabel('Principal Component 2', fontsize=12)
    axes[0].legend(title='Clinical Cohort', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    sns.despine(ax=axes[0])

    # Panel B: ComBat Corrected
    sns.scatterplot(
        data=df_meta, x='ComBat_PC1', y='ComBat_PC2', hue='Cohort', 
        alpha=0.7, s=50, ax=axes[1], palette=palette, legend=False
    )
    axes[1].set_title('B. Post-Harmonization (Empirical Bayes ComBat)', fontsize=14)
    axes[1].set_xlabel('Principal Component 1', fontsize=12)
    axes[1].set_ylabel('Principal Component 2', fontsize=12)
    sns.despine(ax=axes[1])

    plt.tight_layout()
    
    # ---------------------------------------------------------
    # 5. EXPORT
    # ---------------------------------------------------------
    out_path = FIG_OUT / "Fig2A.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[*] SUCCESS! Panel A saved to: {out_path.name}")

if __name__ == "__main__":
    main()