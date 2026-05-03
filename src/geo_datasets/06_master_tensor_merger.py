import os
import pandas as pd
import gc

print("[*] INITIATING QUARANTINE TENSOR MERGER...")

# Directories
base_dir = "/workspace/data"
matrix_dir = os.path.join(base_dir, "processed/mapped_matrices")
labels_path = os.path.join(base_dir, "processed/matrices/master_clinical_labels.csv")
out_dir = os.path.join(base_dir, "processed/ml_tensors")
os.makedirs(out_dir, exist_ok=True)

# THE QUARANTINE ZONE: Lock out the datasets with corrupted biological vocabularies
BLACKLIST = ['GSE63042_mapped.csv.gz']

# 1. Load Labels
print("    -> Loading Master Clinical Labels...")
labels_df = pd.read_csv(labels_path)
labels_df['Patient_ID'] = labels_df['Patient_ID'].astype(str).str.strip()
label_dict = dict(zip(labels_df['Patient_ID'], labels_df['Mortality']))

# 2. Rolling Intersection Tracker (Skipping Blacklist)
matrix_files = sorted([f for f in os.listdir(matrix_dir) if f.endswith('_mapped.csv.gz') and f not in BLACKLIST])
current_intersection = None

print(f"    -> Scanning {len(matrix_files)} pristine mapped matrices for intersection...")

for file in matrix_files:
    file_path = os.path.join(matrix_dir, file)
    df_preview = pd.read_csv(file_path, usecols=[0], index_col=0, compression='gzip')
    genes = set(str(x).strip().upper() for x in df_preview.index.dropna() if str(x).lower() != 'nan')
    
    if current_intersection is None:
        current_intersection = genes
        print(f"    [+] {file}: Started with {len(current_intersection)} genes.")
    else:
        current_intersection = current_intersection.intersection(genes)
        print(f"    [+] {file}: Added. Rolling Core Size -> {len(current_intersection)} genes.")

common_genes = sorted(list(current_intersection))

if len(common_genes) == 0:
    print("[!] ABORTING MERGE: Core gene size is still 0.")
    exit()

print(f"\n    -> SUCCESS: Secured a core genetic signature of {len(common_genes)} genes!")

# 3. Build the Master Tensors
print("    -> Constructing X (Features) and y (Targets) tensors...")
master_X_list = []
master_y_list = []
master_meta_list = []

for file in matrix_files:
    gse_id = file.split('_')[0]
    file_path = os.path.join(matrix_dir, file)
    
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
            
    del df
    gc.collect()

# 4. Finalize and Save
print("\n    -> Finalizing Tensor shapes...")
X_tensor = pd.DataFrame(master_X_list)
y_tensor = pd.DataFrame({'Mortality': master_y_list})
meta_tensor = pd.DataFrame(master_meta_list)

X_tensor = X_tensor.fillna(0)

print("    -> Saving ML Tensors to disk...")
X_tensor.to_csv(os.path.join(out_dir, "X_master.csv.gz"), index=False, compression='gzip')
y_tensor.to_csv(os.path.join(out_dir, "y_master.csv"), index=False)
meta_tensor.to_csv(os.path.join(out_dir, "meta_master.csv"), index=False)

print("\n" + "="*50)
print("[*] TENSOR MERGE COMPLETE.")
print(f"[*] Total Patients: {X_tensor.shape[0]}")
print(f"[*] Total Features (Genes): {X_tensor.shape[1]}")
print(f"[*] X_master shape: {X_tensor.shape}")
print(f"[*] y_master shape: {y_tensor.shape}")
print("="*50)