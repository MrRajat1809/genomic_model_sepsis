"""
01_batch_effect_anova.py

Statistical Batch Effect Validation Module.
Performs one-way Analysis of Variance (ANOVA) on the top Principal Components (PCs) 
of the integrated expression tensor. Statistically evaluates whether technical 
batch effects (Cohort) have been successfully neutralized (p > 0.05) while 
simultaneously ensuring that the core biological signal (Mortality) is preserved (p < 0.05).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway
from sklearn.decomposition import PCA

# Suppress warnings for clean execution logging
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
# Resolves to Multimodal_prediction_sepsis/
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    """Main execution workflow."""
    print("[*] Initiating statistical batch effect validation (ANOVA)...")

    # ---------------------------------------------------------
    # 1. LOAD HARMONIZED DATA
    # ---------------------------------------------------------
    tensor_path = DATA_DIR / "X_atlas.csv.gz"
    if not tensor_path.exists():
        print(f"[ERROR] Required tensor not found: {tensor_path.name}")
        return

    print("    -> Loading integrated expression tensor and metadata...")
    X = pd.read_csv(tensor_path, compression='gzip')
    y = pd.read_csv(DATA_DIR / "y_atlas.csv")
    meta = pd.read_csv(DATA_DIR / "meta_atlas.csv")

    # ---------------------------------------------------------
    # 2. PRINCIPAL COMPONENT ANALYSIS (PCA)
    # ---------------------------------------------------------
    pca = PCA(n_components=3, random_state=42)
    pca_results = pca.fit_transform(X)
    
    df = pd.DataFrame(data=pca_results, columns=['PC1', 'PC2', 'PC3'])
    df['Cohort'] = meta['Dataset']
    df['Mortality'] = y['Mortality']

    explained_var = sum(pca.explained_variance_ratio_) * 100
    print(f"    -> PCA computed. Top 3 components explain {explained_var:.1f}% of total variance.\n")

    # ---------------------------------------------------------
    # 3. STATISTICAL EVALUATION (ANOVA)
    # ---------------------------------------------------------
    cohorts = df['Cohort'].unique()
    
    print("-" * 65)
    print("ANALYSIS I: TECHNICAL BATCH EFFECT (COHORT)")
    print("H0: No significant difference in PC means across cohorts.")
    print("Target: Fail to reject H0 (p > 0.05) -> Variance neutralized.")
    print("-" * 65)
    
    for pc in ['PC1', 'PC2', 'PC3']:
        cohort_groups = [df[df['Cohort'] == c][pc].values for c in cohorts]
        stat, p_val = f_oneway(*cohort_groups)
        
        status = "Neutralized (p > 0.05)" if p_val > 0.05 else "Artifact Detected (p <= 0.05)"
        print(f"  [{pc}] F-statistic: {stat:.3f} | p-value: {p_val:.4e} -> {status}")

    print("\n" + "-" * 65)
    print("ANALYSIS II: BIOLOGICAL SIGNAL (MORTALITY)")
    print("H0: No significant difference in PC means across mortality outcomes.")
    print("Target: Reject H0 (p < 0.05) -> Biological signal preserved.")
    print("-" * 65)
    
    for pc in ['PC1', 'PC2', 'PC3']:
        survivors = df[df['Mortality'] == 0][pc].values
        nonsurvivors = df[df['Mortality'] == 1][pc].values
        stat, p_val = f_oneway(survivors, nonsurvivors)
        
        status = "Preserved (p < 0.05)" if p_val < 0.05 else "Attenuated (p >= 0.05)"
        print(f"  [{pc}] F-statistic: {stat:.3f} | p-value: {p_val:.4e} -> {status}")

    print("\n[*] Statistical validation complete.")

# ==========================================
# EXECUTION GUARD
# ==========================================
if __name__ == "__main__":
    main()