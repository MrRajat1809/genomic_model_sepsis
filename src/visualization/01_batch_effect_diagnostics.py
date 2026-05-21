"""
01_batch_effect_diagnostics.py

Generates the mandatory Batch Effect Correction figure.
Plots PCA of the Raw Log1p integration against the final, 
Gold Standard ComBat-corrected tensor to prove technical 
variance has been neutralized.
"""

import warnings
import gc
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
MATRIX_DIR = BASE_DIR / "data" / "processed" / "mapped_matrices"
ML_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"
FIG_OUT = BASE_DIR / "outputs" / "figures"

BLACKLIST = ['GSE63042_mapped.csv.gz']

def main():
    print("[*] INITIATING COMBAT BATCH EFFECT DIAGNOSTICS...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # 1. Load the ComBat-corrected Master Tensor (The "After")
    print("    -> Loading ComBat-Corrected Master Tensor...")
    X_combat = pd.read_csv(ML_DIR / "X_master.csv.gz", compression='gzip')
    meta = pd.read_csv(ML_DIR / "meta_master.csv")
    common_genes = list(X_combat.columns)

    # 2. Rebuild the Raw Tensor (The "Before")
    print("    -> Rebuilding Raw Tensor for comparison...")
    raw_X = []
    matrix_files = sorted([f.name for f in MATRIX_DIR.iterdir() if f.name.endswith('_mapped.csv.gz') and f.name not in BLACKLIST])
    
    patient_to_cohort = dict(zip(meta['Patient_ID'], meta['Dataset']))

    for file_name in matrix_files:
        df = pd.read_csv(MATRIX_DIR / file_name, index_col=0, compression='gzip')
        df.index = df.index.astype(str).str.strip().str.upper()
        df = df[~df.index.duplicated(keep='first')]
        df = df.loc[common_genes].T
        
        # Log1p the raw data so RNA-Seq doesn't visually crush Microarray
        df_log = np.log1p(df.clip(lower=0)) 
        
        for patient_id in df.index:
            if patient_id in patient_to_cohort:
                raw_X.append(df_log.loc[patient_id].values)
                
        del df, df_log
        gc.collect()

    df_meta = pd.DataFrame({'Cohort': meta['Dataset']})

    # 3. Perform PCA
    print("    -> Calculating Principal Components...")
    pca = PCA(n_components=2, random_state=42)
    
    pca_raw = pca.fit_transform(raw_X)
    pca_combat = pca.fit_transform(X_combat.values)

    df_meta['Raw_PC1'] = pca_raw[:, 0]
    df_meta['Raw_PC2'] = pca_raw[:, 1]
    df_meta['ComBat_PC1'] = pca_combat[:, 0]
    df_meta['ComBat_PC2'] = pca_combat[:, 1]

    # 4. Plotting
    print("    -> Generating Publication Figure...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    sns.set_theme(style="white")
    palette = sns.color_palette("husl", len(df_meta['Cohort'].unique()))

    # Plot A: Before
    sns.scatterplot(data=df_meta, x='Raw_PC1', y='Raw_PC2', hue='Cohort', alpha=0.7, s=50, ax=axes[0], palette=palette)
    axes[0].set_title('A. Before Normalization (Raw Log1p)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Principal Component 1', fontsize=12)
    axes[0].set_ylabel('Principal Component 2', fontsize=12)
    axes[0].legend(title='Dataset', bbox_to_anchor=(1.05, 1), loc='upper left', frameon=False)
    sns.despine(ax=axes[0])

    # Plot B: After ComBat
    sns.scatterplot(data=df_meta, x='ComBat_PC1', y='ComBat_PC2', hue='Cohort', alpha=0.7, s=50, ax=axes[1], palette=palette, legend=False)
    axes[1].set_title('B. After Empirical Bayes (ComBat) Batch Correction', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Principal Component 1', fontsize=12)
    axes[1].set_ylabel('Principal Component 2', fontsize=12)
    sns.despine(ax=axes[1])

    plt.tight_layout()
    save_path = FIG_OUT / "Fig1_ComBat_PCA.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"[*] SUCCESS! ComBat Diagnostics saved to {save_path.name}")

if __name__ == "__main__":
    main()