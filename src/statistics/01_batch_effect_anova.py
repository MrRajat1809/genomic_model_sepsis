"""
01_batch_effect_anova.py

Performs ANOVA on the Principal Components to statistically prove 
that technical batch effects (Cohort) have been neutralized (p > 0.05) 
while biological signals (Mortality) are preserved (p < 0.05).
"""

import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from scipy.stats import f_oneway

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path.cwd().parent if "notebooks" in str(Path.cwd()) else Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"

def main():
    print("[*] INITIATING STATISTICAL BATCH EFFECT VALIDATION (ANOVA)...")

    # 1. Load Data
    X = pd.read_csv(DATA_DIR / "X_master.csv.gz", compression='gzip')
    y = pd.read_csv(DATA_DIR / "y_master.csv")
    meta = pd.read_csv(DATA_DIR / "meta_master.csv")

    # 2. Run PCA
    pca = PCA(n_components=3, random_state=42)
    pca_results = pca.fit_transform(X)
    
    df = pd.DataFrame(data=pca_results, columns=['PC1', 'PC2', 'PC3'])
    df['Cohort'] = meta['Dataset']
    df['Mortality'] = y['Mortality']

    print(f"    -> PCA computed. Top 3 components explain {sum(pca.explained_variance_ratio_)*100:.1f}% of total variance.\n")

    # 3. Perform ANOVA
    cohorts = df['Cohort'].unique()
    
    print("=== ANOVA RESULTS: BATCH EFFECT (COHORT) ===")
    print("Hypothesis: Cohorts are statistically different. (We want to REJECT this, so we want p > 0.05)")
    for pc in ['PC1', 'PC2', 'PC3']:
        cohort_groups = [df[df['Cohort'] == c][pc].values for c in cohorts]
        stat, p_val = f_oneway(*cohort_groups)
        status = "PASSED (No Batch Effect)" if p_val > 0.05 else "FAILED (Batch Effect Exists)"
        print(f"  - {pc}: F-stat = {stat:.3f}, p-value = {p_val:.4e} -> {status}")

    print("\n=== ANOVA RESULTS: BIOLOGICAL EFFECT (MORTALITY) ===")
    print("Hypothesis: Survivors and Non-Survivors are statistically different. (We want to ACCEPT this, so we want p < 0.05)")
    for pc in ['PC1', 'PC2', 'PC3']:
        survivors = df[df['Mortality'] == 0][pc].values
        nonsurvivors = df[df['Mortality'] == 1][pc].values
        stat, p_val = f_oneway(survivors, nonsurvivors)
        status = "PASSED (Biology Preserved)" if p_val < 0.05 else "FAILED (Signal Lost)"
        print(f"  - {pc}: F-stat = {stat:.3f}, p-value = {p_val:.4e} -> {status}")

if __name__ == "__main__":
    main()