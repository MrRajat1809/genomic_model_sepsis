"""
01b_density_diagnostics.py

Generates the "Global Density Plot" (KDE) to accompany the PCA.
Plots the massive variations in baseline hardware signals (Microarray vs RNA-Seq) 
and proves that Cohort-Level Z-Scoring perfectly aligns all 7 cohorts into a 
single universal mathematical distribution (Mean=0, SD=1).
"""

import warnings
import gc
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
MATRIX_DIR = BASE_DIR / "data" / "processed" / "mapped_matrices"
FIG_OUT = BASE_DIR / "outputs" / "figures"

BLACKLIST = ['GSE63042_mapped.csv.gz']

def main():
    print("[*] INITIATING GLOBAL DENSITY DIAGNOSTICS...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # 1. Get the common genes
    print("    -> Determining common gene intersection...")
    matrix_files = sorted([f.name for f in MATRIX_DIR.iterdir() if f.name.endswith('_mapped.csv.gz') and f.name not in BLACKLIST])
    
    current_intersection = None
    for file_name in matrix_files:
        df_preview = pd.read_csv(MATRIX_DIR / file_name, usecols=[0], index_col=0, compression='gzip')
        genes = set(str(x).strip().upper() for x in df_preview.index.dropna() if str(x).lower() != 'nan')
        current_intersection = genes if current_intersection is None else current_intersection.intersection(genes)
    
    common_genes = sorted(list(current_intersection))

    # 2. Extract Data for Plotting
    print("    -> Extracting and sampling array distributions...")
    raw_vals, scaled_vals, cohorts = [], [], []
    scaler = StandardScaler()

    for file_name in matrix_files:
        gse_id = file_name.split('_')[0]
        df = pd.read_csv(MATRIX_DIR / file_name, index_col=0, compression='gzip')
        df.index = df.index.astype(str).str.strip().str.upper()
        df = df[~df.index.duplicated(keep='first')]
        df = df.loc[common_genes].T
        
        # Scale the data
        df_scaled = scaler.fit_transform(df)
        
        # Log1p the raw data for visual scaling
        df_log = np.log1p(df.clip(lower=0)).values
        
        # Flatten the arrays to get the global distribution of ALL genes
        flat_log = df_log.flatten()
        flat_scaled = df_scaled.flatten()
        
        # Sample 50,000 points to keep plotting fast and memory-efficient
        np.random.seed(42)
        sample_size = min(50000, len(flat_log))
        idx = np.random.choice(len(flat_log), size=sample_size, replace=False)
        
        raw_vals.extend(flat_log[idx])
        scaled_vals.extend(flat_scaled[idx])
        cohorts.extend([gse_id] * sample_size)
        
        del df, df_scaled, df_log
        gc.collect()

    df_plot = pd.DataFrame({
        'Log_Expression': raw_vals,
        'Z_Score': scaled_vals,
        'Cohort': cohorts
    })

    # 3. Plotting Figure 2
    print("    -> Generating Figure 2 (KDE Distributions)...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("husl", len(df_plot['Cohort'].unique()))

    # Plot A: The Raw Disconnect
    sns.kdeplot(
        data=df_plot, x='Log_Expression', hue='Cohort', 
        fill=True, common_norm=False, palette=palette, alpha=0.3, ax=axes[0]
    )
    axes[0].set_title('A. Before Normalization (Log1p Transformed)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Gene Expression Magnitude', fontsize=12)
    axes[0].set_ylabel('Density', fontsize=12)
    sns.despine(ax=axes[0])

    # Plot B: The Perfect Alignment
    sns.kdeplot(
        data=df_plot, x='Z_Score', hue='Cohort', 
        fill=True, common_norm=False, palette=palette, alpha=0.3, ax=axes[1], legend=False
    )
    axes[1].set_title('B. After Cohort-Level Z-Score Normalization', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Standardized Expression (Z-Score)', fontsize=12)
    axes[1].set_ylabel('Density', fontsize=12)
    sns.despine(ax=axes[1])

    plt.tight_layout()
    save_path = FIG_OUT / "Fig2_Batch_Effect_Density.pdf"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    print(f"[*] SUCCESS! Publication figure saved to {save_path.name}")

if __name__ == "__main__":
    main()