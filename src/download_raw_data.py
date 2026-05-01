import os
import gc
import urllib.request
import pandas as pd
import numpy as np
import GEOparse
from sklearn.ensemble import IsolationForest
from neuroCombat import neuroCombat
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

print("[*] INITIATING SMART GEO HARVESTER & NORMALIZATION PIPELINE...")

# 1. Define the targets (Cleaned List)
target_datasets = [
    'GSE65682', 'GSE54514', 'GSE95233', 'GSE28750', 
    'GSE74224', 'GSE11375', 'GSE66099', 'GSE26440' 
]
download_dir = "/workspace/data/raw/geo_pool"
processed_dir = "/workspace/data/processed/geo_pool"
os.makedirs(download_dir, exist_ok=True)
os.makedirs(processed_dir, exist_ok=True)

processed_dataframes = []
batch_labels = []

# --- THE VISUAL DOWNLOAD BYPASS ---
class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def secure_download_geo(gse_id, dest_dir):
    """Bypasses size bug and displays a live progress bar."""
    prefix = gse_id[:-3] + "nnn"
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse_id}/soft/{gse_id}_family.soft.gz"
    file_path = os.path.join(dest_dir, f"{gse_id}_family.soft.gz")
    
    if not os.path.exists(file_path):
        print(f"    -> Force-downloading {gse_id} over HTTPS...")
        try:
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=gse_id) as t:
                urllib.request.urlretrieve(url, file_path, reporthook=t.update_to)
        except Exception as e:
            print(f"    [!] HTTPS failed, falling back to FTP: {e}")
            ftp_url = url.replace("https://", "ftp://")
            with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=gse_id+" (FTP)") as t:
                urllib.request.urlretrieve(ftp_url, file_path, reporthook=t.update_to)
    else:
        print(f"    -> File {gse_id}_family.soft.gz already exists. Skipping download.")
        
    return file_path
# ----------------------------------

# 2. Extraction & Translation Loop
for batch_idx, gse_id in enumerate(target_datasets):
    print(f"\n[+] Processing Cohort: {gse_id} (Batch {batch_idx})...")
    try:
        local_filepath = secure_download_geo(gse_id, download_dir)
        
        print(f"    -> Unpacking and parsing the massive text matrix (This takes 2-5 minutes)...")
        gse = GEOparse.get_GEO(filepath=local_filepath, silent=True)
        
        gpl_name = list(gse.gpls.keys())[0]
        gpl_table = gse.gpls[gpl_name].table
        
        # Smart Column Hunter
        search_terms = ['symbol', 'gene_assignment', 'gene_name', 'orf', 'ilmn_gene']
        symbol_col = next((col for col in gpl_table.columns if any(x in col.lower() for x in search_terms)), None)
        if not symbol_col:
            print(f"[!] No Symbol column for {gse_id}. Skipping.")
            continue
            
        probe_to_symbol = dict(zip(gpl_table['ID'].astype(str), gpl_table[symbol_col].astype(str)))
        
        patient_data = {}
        for gsm_name, gsm in gse.gsms.items():
            expr = gsm.table.set_index('ID_REF')['VALUE']
            expr.index = expr.index.astype(str).map(probe_to_symbol)
            expr = expr.loc[expr.index.notna() & (expr.index != 'nan') & (expr.index != '')]
            
            # --- THE FORMATTING FIX ---
            # Standardize to uppercase and remove trailing spaces to prevent dropping valid genes
            expr.index = expr.index.str.upper().str.strip()
            
            expr = expr.groupby(expr.index).max()
            patient_data[gsm_name] = expr

        hospital_df = pd.DataFrame(patient_data)
        print(f"    -> Harvested {hospital_df.shape[1]} patients, {hospital_df.shape[0]} unique genes.")
        
        processed_dataframes.append(hospital_df)
        batch_labels.extend([batch_idx] * hospital_df.shape[1])
        
    except Exception as e:
        print(f"[!] Error processing {gse_id}: {e}")
        
    finally:
        if 'gse' in locals(): del gse
        if 'patient_data' in locals(): del patient_data
        if 'hospital_df' in locals(): del hospital_df
        gc.collect()


# 3. The Progressive Intersection (Finding the Rogue Dataset)
print("\n[*] Executing Progressive Intersection...")
if not processed_dataframes:
    raise ValueError("No dataframes were successfully processed!")

master_matrix = processed_dataframes[0]
valid_batch_labels = [batch_labels[i] for i in range(master_matrix.shape[1])]

current_patient_idx = master_matrix.shape[1]

for i, df in enumerate(processed_dataframes[1:]):
    print(f"    -> Merging Dataframe {i+2} of {len(processed_dataframes)}...")
    
    # Check the overlap before merging
    overlap = len(master_matrix.index.intersection(df.index))
    if overlap < 1000:
        print(f"    [!!!] WARNING: This batch only shares {overlap} genes with the master! Skipping to protect matrix.")
        # We must skip its batch labels as well
        current_patient_idx += df.shape[1]
        continue
        
    master_matrix = master_matrix.join(df, how='inner')
    
    # Add the batch labels only for the patients we successfully merged
    patients_in_this_batch = df.shape[1]
    valid_batch_labels.extend(batch_labels[current_patient_idx : current_patient_idx + patients_in_this_batch])
    current_patient_idx += patients_in_this_batch
    
    print(f"    -> Genes remaining in Data Lake: {master_matrix.shape[0]}")

master_matrix.dropna(inplace=True)
# Ensure batch labels match the final master matrix length
batch_labels = valid_batch_labels

print(f"\n    -> Pre-Outlier Dimensions: {master_matrix.shape[1]} patients, {master_matrix.shape[0]} genes.")


# 4. Outlier Detection
print("\n[*] Hunting for Patient Outliers (Isolation Forest)...")
X_outlier_check = master_matrix.T.values 
iso_forest = IsolationForest(contamination=0.05, random_state=42) 
outlier_predictions = iso_forest.fit_predict(X_outlier_check)

valid_patient_mask = outlier_predictions == 1
clean_matrix = master_matrix.loc[:, valid_patient_mask]
clean_batch_labels = [label for i, label in enumerate(batch_labels) if valid_patient_mask[i]]

num_outliers = sum(outlier_predictions == -1)
print(f"    -> Removed {num_outliers} corrupted/outlier patient samples.")


# 5. ComBat Normalization (The Final Boss)
print("\n[*] Executing neuroCombat (Erasing Hardware Bias)...")
# neuroCombat expects genes as rows, patients as columns
data_for_combat = clean_matrix.values

# Create a formal Covariates DataFrame for neuroCombat
covars_df = pd.DataFrame({'batch_id': clean_batch_labels})

# Run the algorithm using the updated syntax
combat_results = neuroCombat(dat=data_for_combat, covars=covars_df, batch_col='batch_id')
corrected_matrix_values = combat_results['data']

# Rebuild the DataFrame
final_corrected_df = pd.DataFrame(corrected_matrix_values, 
                                  index=clean_matrix.index, 
                                  columns=clean_matrix.columns)

print("\n======================================================")
print(f"FINAL BATCH-CORRECTED DATA LAKE DIMENSIONS:")
print(f"Total Clean Patients: {final_corrected_df.shape[1]}")
print(f"Universal Genes: {final_corrected_df.shape[0]}")
print("======================================================")

# Save the pristine, corrected Data Lake
save_path = os.path.join(processed_dir, "master_genomic_matrix_combat_corrected.csv")
final_corrected_df.to_csv(save_path)
print(f"[+] Pristine Master Matrix safely written to disk: {save_path}")