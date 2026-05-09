"""
06_master_tensor_merger.py

Quarantine Tensor Merger.
Iterates through mapped gene expression matrices, identifies the intersection 
of common genes across all valid cohorts (excluding blacklisted datasets), 
and constructs the final X (features) and y (targets) tensors for machine learning.
Outputs compressed, ready-to-train ML tensors and metadata trackers.
"""

import gc
from pathlib import Path
import pandas as pd

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
# Dynamically resolve paths from the script's location (src/geo_datasets/)
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
MATRIX_DIR = DATA_DIR / "processed" / "mapped_matrices"
LABELS_PATH = DATA_DIR / "processed" / "matrices" / "master_clinical_labels.csv"
OUT_DIR = DATA_DIR / "processed" / "ml_tensors"

# THE QUARANTINE ZONE: Lock out the datasets with corrupted biological vocabularies
BLACKLIST = ['GSE63042_mapped.csv.gz']

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    print("[*] INITIATING QUARANTINE TENSOR MERGER...")

    # Ensure output directory exists
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load Labels
    print("    -> Loading Master Clinical Labels...")
    if not LABELS_PATH.exists():
        print(f"[!] Critical Error: {LABELS_PATH.name} not found. Run the matrix parser first.")
        return

    labels_df = pd.read_csv(LABELS_PATH)
    labels_df['Patient_ID'] = labels_df['Patient_ID'].astype(str).str.strip()
    label_dict = dict(zip(labels_df['Patient_ID'], labels_df['Mortality']))

    # 2. Rolling Intersection Tracker (Skipping Blacklist)
    # Filter for mapped matrices and exclude blacklisted files
    matrix_files = sorted([
        f.name for f in MATRIX_DIR.iterdir() 
        if f.name.endswith('_mapped.csv.gz') and f.name not in BLACKLIST
    ])
    
    if not matrix_files:
        print(f"[!] No valid mapped matrices found in {MATRIX_DIR.name}.")
        return

    current_intersection = None

    print(f"    -> Scanning {len(matrix_files)} pristine mapped matrices for intersection...")

    for file_name in matrix_files:
        file_path = MATRIX_DIR / file_name
        df_preview = pd.read_csv(file_path, usecols=[0], index_col=0, compression='gzip')
        genes = set(str(x).strip().upper() for x in df_preview.index.dropna() if str(x).lower() != 'nan')
        
        if current_intersection is None:
            current_intersection = genes
            print(f"    [+] {file_name}: Started with {len(current_intersection)} genes.")
        else:
            current_intersection = current_intersection.intersection(genes)
            print(f"    [+] {file_name}: Added. Rolling Core Size -> {len(current_intersection)} genes.")

    common_genes = sorted(list(current_intersection))

    if len(common_genes) == 0:
        print("[!] ABORTING MERGE: Core gene size is 0. No overlapping genes found.")
        return

    print(f"\n    -> SUCCESS: Secured a core genetic signature of {len(common_genes)} genes!")

    # 3. Build the Master Tensors
    print("    -> Constructing X (Features) and y (Targets) tensors...")
    master_X_list = []
    master_y_list = []
    master_meta_list = []

    for file_name in matrix_files:
        gse_id = file_name.split('_')[0]
        file_path = MATRIX_DIR / file_name
        
        df = pd.read_csv(file_path, index_col=0, compression='gzip')
        df.index = df.index.astype(str).str.strip().str.upper()
        df.columns = df.columns.astype(str).str.strip()
        df = df[~df.index.duplicated(keep='first')]
        
        # Slice down to ONLY the common genes
        df = df.loc[common_genes]
        df = df.T
        
        matched_patients = 0
        for patient_id, gene_expression in df.iterrows():
            if patient_id in label_dict:
                master_X_list.append(gene_expression)
                master_y_list.append(label_dict[patient_id])
                master_meta_list.append({'Dataset': gse_id, 'Patient_ID': patient_id})
                matched_patients += 1
                
        print(f"        -> {gse_id}: Mapped {matched_patients} labeled patients.")
                
        # Free RAM
        del df
        gc.collect()

    # 4. Finalize and Save
    print("\n    -> Finalizing Tensor shapes...")
    X_tensor = pd.DataFrame(master_X_list)
    y_tensor = pd.DataFrame({'Mortality': master_y_list})
    meta_tensor = pd.DataFrame(master_meta_list)

    X_tensor = X_tensor.fillna(0)

    print("    -> Saving ML Tensors to disk...")
    X_tensor.to_csv(OUT_DIR / "X_master.csv.gz", index=False, compression='gzip')
    y_tensor.to_csv(OUT_DIR / "y_master.csv", index=False)
    meta_tensor.to_csv(OUT_DIR / "meta_master.csv", index=False)

    print("\n" + "="*50)
    print("[*] TENSOR MERGE COMPLETE.")
    print(f"[*] Total Patients: {X_tensor.shape[0]}")
    print(f"[*] Total Features (Genes): {X_tensor.shape[1]}")
    print(f"[*] X_master shape: {X_tensor.shape}")
    print(f"[*] y_master shape: {y_tensor.shape}")
    print("="*50)

if __name__ == "__main__":
    main()