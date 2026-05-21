"""
06_master_tensor_merger.py

Gold Standard Integrative Tensor Merger.
1. Identifies intersecting genes across all pristine cohorts.
2. Applies within-cohort Z-scoring to standardize expression scales.
3. Merges the cohorts into a master tensor.
4. Applies ComBat (Empirical Bayes) to subtract latent technical batch effects 
   while strictly preserving the 'Mortality' biological covariate.
"""

import gc
import warnings
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from neuroCombat import neuroCombat

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
MATRIX_DIR = DATA_DIR / "processed" / "mapped_matrices"
LABELS_PATH = DATA_DIR / "processed" / "matrices" / "master_clinical_labels.csv"
OUT_DIR = DATA_DIR / "processed" / "ml_tensors"

BLACKLIST = ['GSE63042_mapped.csv.gz']

def main():
    print("[*] INITIATING GOLD STANDARD TENSOR MERGER (ComBat + Z-Score)...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Labels
    print("    -> Loading Master Clinical Labels...")
    if not LABELS_PATH.exists():
        print(f"[!] Critical Error: {LABELS_PATH.name} not found.")
        return

    labels_df = pd.read_csv(LABELS_PATH)
    labels_df['Patient_ID'] = labels_df['Patient_ID'].astype(str).str.strip()
    label_dict = dict(zip(labels_df['Patient_ID'], labels_df['Mortality']))

    # 2. Find Intersecting Genes
    matrix_files = sorted([f.name for f in MATRIX_DIR.iterdir() if f.name.endswith('_mapped.csv.gz') and f.name not in BLACKLIST])
    if not matrix_files: return

    current_intersection = None
    for file_name in matrix_files:
        df_preview = pd.read_csv(MATRIX_DIR / file_name, usecols=[0], index_col=0, compression='gzip')
        genes = set(str(x).strip().upper() for x in df_preview.index.dropna() if str(x).lower() != 'nan')
        current_intersection = genes if current_intersection is None else current_intersection.intersection(genes)

    common_genes = sorted(list(current_intersection))
    print(f"    -> SUCCESS: Secured core signature of {len(common_genes)} genes.")

    # 3. Z-Score and Merge
    print("    -> Constructing and Z-Scoring within-cohort matrices...")
    master_df_list = []
    covariates_list = []
    scaler = StandardScaler()

    for file_name in matrix_files:
        gse_id = file_name.split('_')[0]
        df = pd.read_csv(MATRIX_DIR / file_name, index_col=0, compression='gzip')
        df.index = df.index.astype(str).str.strip().str.upper()
        df.columns = df.columns.astype(str).str.strip()
        df = df[~df.index.duplicated(keep='first')].loc[common_genes].T
        
        # Within-Cohort Z-Scoring
        scaled_values = scaler.fit_transform(df)
        df_scaled = pd.DataFrame(scaled_values, index=df.index, columns=df.columns)
        
        for patient_id, gene_expression in df_scaled.iterrows():
            if patient_id in label_dict:
                master_df_list.append(gene_expression)
                covariates_list.append({
                    'Patient_ID': patient_id,
                    'Batch': gse_id,
                    'Mortality': label_dict[patient_id]
                })
        
        del df, df_scaled
        gc.collect()

    # 4. Prepare for ComBat
    print("\n    -> Initializing ComBat Batch Correction...")
    
    # neuroCombat expects data as Genes x Patients (so we transpose)
    raw_merged_tensor = pd.DataFrame(master_df_list).T 
    covars_df = pd.DataFrame(covariates_list)
    
    print(f"       - Total Patients: {raw_merged_tensor.shape[1]}")
    print(f"       - Total Genes: {raw_merged_tensor.shape[0]}")
    print(f"       - Batches Detected: {covars_df['Batch'].nunique()}")

    # 5. Run Empirical Bayes ComBat
    # We MUST protect 'Mortality' as a categorical variable so ComBat doesn't erase the disease signal!
    print("    -> Executing Empirical Bayes modeling (This may take a moment)...")
    combat_results = neuroCombat(
        dat=raw_merged_tensor.values,
        covars=covars_df,
        batch_col='Batch',
        categorical_cols=['Mortality']
    )
    
    # 6. Finalize Tensors
    print("    -> Finalizing Batch-Corrected Tensors...")
    # Re-transpose back to Patients x Genes for Machine Learning
    X_combat = pd.DataFrame(combat_results['data'].T, columns=common_genes, index=covars_df['Patient_ID'])
    y_tensor = covars_df[['Mortality']]
    meta_tensor = covars_df[['Batch', 'Patient_ID']].rename(columns={'Batch': 'Dataset'})

    print("    -> Saving Tensors to disk...")
    X_combat.to_csv(OUT_DIR / "X_master.csv.gz", index=False, compression='gzip')
    y_tensor.to_csv(OUT_DIR / "y_master.csv", index=False)
    meta_tensor.to_csv(OUT_DIR / "meta_master.csv", index=False)

    print("\n" + "="*50)
    print("[*] GOLD STANDARD TENSOR MERGE COMPLETE.")
    print("="*50)

if __name__ == "__main__":
    main()