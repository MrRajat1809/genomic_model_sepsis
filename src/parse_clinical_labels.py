import os
import gc
import pandas as pd
import numpy as np
import GEOparse
import warnings
warnings.filterwarnings("ignore")

print("[*] INITIATING GEO CLINICAL METADATA PARSER...")

raw_dir = "/workspace/data/raw/geo_pool"
processed_dir = "/workspace/data/processed/geo_pool"
matrix_path = os.path.join(processed_dir, "master_genomic_matrix_combat_corrected.csv")

print("[*] Loading Master Genomic Matrix to identify our target patients...")
df_matrix = pd.read_csv(matrix_path, index_col=0)
target_patients = set(df_matrix.columns)
print(f"    -> Found {len(target_patients)} patients needing survival labels.")

target_datasets = [
    'GSE65682', 'GSE54514', 'GSE95233', 'GSE28750', 
    'GSE74224', 'GSE11375', 'GSE66099', 'GSE26440' 
]

labels_dict = {}

for gse_id in target_datasets:
    filepath = os.path.join(raw_dir, f"{gse_id}_family.soft.gz")
    if not os.path.exists(filepath):
        continue
        
    print(f"\n[+] Mining metadata from {gse_id}...")
    gse = GEOparse.get_GEO(filepath=filepath, silent=True)
    
    found_in_gse = 0
    for gsm_name, gsm in gse.gsms.items():
        # Only parse patients that survived our Outlier/ComBat filtering
        if gsm_name not in target_patients:
            continue
            
        mortality_label = np.nan
        
        # The Regex Hunter: Look through all characteristics
        characteristics = gsm.metadata.get('characteristics_ch1', [])
        source_name = gsm.metadata.get('source_name_ch1', [''])[0].lower()
        title = gsm.metadata.get('title', [''])[0].lower()
        
        # 1. Healthy Control check
        if any(kw in source_name for kw in ['healthy', 'control']) or any(kw in title for kw in ['healthy', 'control']):
            mortality_label = 0
        else:
            # 2. Sepsis Survival check
            for char in characteristics:
                char_lower = char.lower()
                
                # Split into key and value (e.g., "mortality_event_28days" : "0")
                if ':' in char_lower:
                    key, val = char_lower.split(':', 1)
                    key = key.strip()
                    val = val.strip()
                    
                    # Target the exact column that mentions survival
                    if any(kw in key for kw in ['mortality', 'survival', 'outcome', 'status', 'death']):
                        
                        # Positive (Dead) signatures
                        if val in ['1', 'yes', 'dead', 'death', 'non-survivor', 'died', 'deceased']:
                            mortality_label = 1
                            break
                        
                        # Negative (Alive) signatures
                        elif val in ['0', 'no', 'alive', 'survivor', 'survived']:
                            mortality_label = 0
                            break
        
        # Log the label if we successfully parsed it
        if not pd.isna(mortality_label):
            labels_dict[gsm_name] = mortality_label
            found_in_gse += 1
            
    print(f"    -> Successfully extracted {found_in_gse} labels.")
    
    # Memory management
    del gse
    gc.collect()

# Convert our findings into a strict DataFrame
df_labels = pd.DataFrame.from_dict(labels_dict, orient='index', columns=['mortality'])
print(f"\n[*] PARSING COMPLETE. Total Real Labels Found: {len(df_labels)}")

# The Synchronizer: Drop patients we couldn't find labels for
print("\n[*] Synchronizing Master Matrix with Labeled Patients...")
valid_patients = df_labels.index.tolist()
df_matrix_labeled = df_matrix[valid_patients]

print(f"    -> Final Synced Dimensions: {df_matrix_labeled.shape[1]} Patients, {df_matrix_labeled.shape[0]} Genes")

# Save outputs (These are the actual files we will feed XGBoost)
matrix_out_path = os.path.join(processed_dir, "X_matrix_labeled.csv")
labels_out_path = os.path.join(processed_dir, "y_labels.csv")

df_matrix_labeled.to_csv(matrix_out_path)
df_labels.to_csv(labels_out_path)

print(f"[+] Ground-Truth Matrix safely written to disk: {matrix_out_path}")
print(f"[+] Ground-Truth Labels safely written to disk: {labels_out_path}")