"""
05_master_tensor_merger.py

Integrative Tensor Merger, Batch Correction, and ML Quarantine Protocol.
Performs the following dual-pipeline operations:
1. Identifies intersecting gene features across all processed cohorts.
2. Applies within-cohort Z-scoring to standardize expression scales.
3. Biological Pipeline: Generates 'X_atlas' (all cohorts globally ComBat-corrected).
4. Machine Learning Pipeline: 
   - Generates 'X_train' (6 cohorts locally ComBat-corrected).
   - Generates 'X_vault' (1 cohort strictly Z-scored, bypassing ComBat to prevent data leakage).
"""

import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from neuroCombat import neuroCombat
from sklearn.preprocessing import StandardScaler

# Suppress warnings for clean execution logging
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
# Resolves to Multimodal_prediction_sepsis/
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
MATRIX_DIR = DATA_DIR / "processed" / "mapped_matrices"
LABELS_PATH = DATA_DIR / "processed" / "matrices" / "master_clinical_labels.csv"
OUT_DIR = DATA_DIR / "processed" / "ml_tensors"

BLACKLIST = ['GSE63042_mapped.csv.gz']
HOLDOUT_COHORT = 'GSE65682'

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    """Main execution workflow."""
    print("[*] Initiating dual-pipeline tensor merger and quarantine protocol...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD MASTER CLINICAL LABELS
    # ---------------------------------------------------------
    print("    -> Loading global clinical labels...")
    if not LABELS_PATH.exists():
        print(f"[ERROR] Required label registry not found: {LABELS_PATH.name}")
        return

    labels_df = pd.read_csv(LABELS_PATH)
    labels_df['Patient_ID'] = labels_df['Patient_ID'].astype(str).str.strip()
    label_dict = dict(zip(labels_df['Patient_ID'], labels_df['Mortality']))

    # ---------------------------------------------------------
    # 2. IDENTIFY INTERSECTING GENE FEATURES
    # ---------------------------------------------------------
    matrix_files = sorted([f.name for f in MATRIX_DIR.iterdir() if f.name.endswith('_mapped.csv.gz') and f.name not in BLACKLIST])
    if not matrix_files:
        print("[ERROR] No mapped matrices found for integration.")
        return

    current_intersection = None
    for file_name in matrix_files:
        df_preview = pd.read_csv(MATRIX_DIR / file_name, usecols=[0], index_col=0, compression='gzip')
        genes = set(str(x).strip().upper() for x in df_preview.index.dropna() if str(x).lower() != 'nan')
        current_intersection = genes if current_intersection is None else current_intersection.intersection(genes)

    common_genes = sorted(list(current_intersection))
    print(f"    [+] Core feature intersection secured: {len(common_genes)} universal genes.")

    # ---------------------------------------------------------
    # 3. WITHIN-COHORT STANDARDIZATION & BIFURCATION
    # ---------------------------------------------------------
    print("    -> Constructing and Z-Scoring within-cohort matrices...")
    
    # Atlas Lists (All 7 Cohorts)
    df_atlas, covars_atlas = [], []
    
    # ML Train Lists (6 Cohorts)
    df_train, covars_train = [], []
    
    # ML Vault Lists (1 Cohort)
    df_vault, covars_vault = [], []

    scaler = StandardScaler()

    for file_name in matrix_files:
        gse_id = file_name.split('_')[0]
        df = pd.read_csv(MATRIX_DIR / file_name, index_col=0, compression='gzip')
        
        # Standardize indices and dimensions
        df.index = df.index.astype(str).str.strip().str.upper()
        df.columns = df.columns.astype(str).str.strip()
        df = df[~df.index.duplicated(keep='first')].loc[common_genes].T
        
        # Apply strict within-cohort Z-Scoring
        scaled_values = scaler.fit_transform(df)
        df_scaled = pd.DataFrame(scaled_values, index=df.index, columns=df.columns)
        
        # Align with clinical labels and bifurcate paths
        for patient_id, gene_expression in df_scaled.iterrows():
            if patient_id in label_dict:
                meta_entry = {
                    'Patient_ID': patient_id,
                    'Dataset': gse_id,
                    'Mortality': label_dict[patient_id]
                }
                
                # Always add to Atlas
                df_atlas.append(gene_expression)
                covars_atlas.append(meta_entry)
                
                # Route to Vault or Mega-Train
                if gse_id == HOLDOUT_COHORT:
                    df_vault.append(gene_expression)
                    covars_vault.append(meta_entry)
                else:
                    df_train.append(gene_expression)
                    covars_train.append(meta_entry)
        
        del df, df_scaled
        gc.collect()

    # ---------------------------------------------------------
    # 4. EXECUTE COMBAT ON 'ATLAS' (BIOLOGICAL PROOF)
    # ---------------------------------------------------------
    print("\n    -> [PIPELINE A] Initializing Global Atlas ComBat (7 Cohorts)...")
    covars_atlas_df = pd.DataFrame(covars_atlas)
    raw_atlas_tensor = pd.DataFrame(df_atlas).T 
    
    combat_atlas_res = neuroCombat(
        dat=raw_atlas_tensor.values,
        covars=covars_atlas_df,
        batch_col='Dataset',
        categorical_cols=['Mortality']
    )
    X_atlas = pd.DataFrame(combat_atlas_res['data'].T, columns=common_genes, index=covars_atlas_df['Patient_ID'])

    # ---------------------------------------------------------
    # 5. EXECUTE COMBAT ON 'MEGA-TRAIN' (MACHINE LEARNING)
    # ---------------------------------------------------------
    print("\n    -> [PIPELINE B] Initializing Mega-Train ComBat (6 Cohorts)...")
    covars_train_df = pd.DataFrame(covars_train)
    raw_train_tensor = pd.DataFrame(df_train).T 
    
    combat_train_res = neuroCombat(
        dat=raw_train_tensor.values,
        covars=covars_train_df,
        batch_col='Dataset',
        categorical_cols=['Mortality']
    )
    X_train = pd.DataFrame(combat_train_res['data'].T, columns=common_genes, index=covars_train_df['Patient_ID'])

    # ---------------------------------------------------------
    # 6. FINALIZE 'VAULT' (STRICT QUARANTINE)
    # ---------------------------------------------------------
    print("\n    -> [PIPELINE C] Finalizing Locked Vault (Z-Scored ONLY)...")
    covars_vault_df = pd.DataFrame(covars_vault)
    X_vault = pd.DataFrame(df_vault, columns=common_genes, index=covars_vault_df['Patient_ID'])

    # ---------------------------------------------------------
    # 7. EXPORT TENSORS
    # ---------------------------------------------------------
    print("\n    -> Exporting Tensors to disk...")
    
    # Export Atlas (For PCA/Zenodo)
    X_atlas.to_csv(OUT_DIR / "X_atlas.csv.gz", index=False, compression='gzip')
    covars_atlas_df[['Mortality']].to_csv(OUT_DIR / "y_atlas.csv", index=False)
    covars_atlas_df[['Dataset', 'Patient_ID']].to_csv(OUT_DIR / "meta_atlas.csv", index=False)

    # Export Mega-Train (For ML Benchmarking)
    X_train.to_csv(OUT_DIR / "X_train.csv.gz", index=False, compression='gzip')
    covars_train_df[['Mortality']].to_csv(OUT_DIR / "y_train.csv", index=False)
    covars_train_df[['Dataset', 'Patient_ID']].to_csv(OUT_DIR / "meta_train.csv", index=False)

    # Export Vault (For Final External Validation)
    X_vault.to_csv(OUT_DIR / "X_vault.csv.gz", index=False, compression='gzip')
    covars_vault_df[['Mortality']].to_csv(OUT_DIR / "y_vault.csv", index=False)
    covars_vault_df[['Dataset', 'Patient_ID']].to_csv(OUT_DIR / "meta_vault.csv", index=False)

    print("\n" + "=" * 50)
    print("[*] Dual-pipeline tensor generation complete.")
    print(f"    - Atlas Patients : {len(covars_atlas_df)}")
    print(f"    - Train Patients : {len(covars_train_df)}")
    print(f"    - Vault Patients : {len(covars_vault_df)}")
    print("=" * 50)

# ==========================================
# EXECUTION GUARD
# ==========================================
if __name__ == "__main__":
    main()