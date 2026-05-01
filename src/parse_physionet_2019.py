import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

# 1. Set your paths
data_dir = "/workspace/data/raw/physionet_2019/physionet.org/files/challenge-2019/1.0.0/training/"
output_dir = "/workspace/data/processed/physionet_pool/"

os.makedirs(output_dir, exist_ok=True)
patient_files = glob.glob(os.path.join(data_dir, "**/*.psv"), recursive=True)
print(f"[*] Found {len(patient_files)} patient files to process.")

processed_patients = []

# 2. Parse the files with strict clinical rules
print("[*] Parsing first-24-hour time-series data...")
for file in tqdm(patient_files):
    df = pd.read_csv(file, sep='|')
    patient_id = os.path.basename(file).split('.')[0]
    
    # RULE 1: Exclude patients who stayed in the ICU for less than 24 hours
    if len(df) < 24:
        continue
    
    # The Label: Did they EVER develop sepsis during their stay?
    # We look at the whole file to determine the final truth.
    is_sepsis = df['SepsisLabel'].max() 
    
    # RULE 2: The 24-Hour Window (Preventing Target Leakage)
    # We slice the dataframe to ONLY look at the first 24 hours (first 24 rows)
    first_24h_df = df.head(24)
    
    features_df = first_24h_df.drop(columns=['SepsisLabel', 'Unit1', 'Unit2'])
    
    # Calculate stats ONLY on those first 24 hours
    summary = features_df.agg(['mean', 'min', 'max']).unstack()
    summary.index = [f"{col}_{stat}" for col, stat in summary.index]
    
    summary['Patient_ID'] = patient_id
    summary['Sepsis_Outcome'] = is_sepsis
    
    processed_patients.append(summary)

# 3. Assemble Master Matrix
print("\n[*] Assembling Master Clinical Matrix...")
master_df = pd.DataFrame(processed_patients)

cols = ['Patient_ID', 'Sepsis_Outcome'] + [c for c in master_df.columns if c not in ['Patient_ID', 'Sepsis_Outcome']]
master_df = master_df[cols]
master_df.set_index('Patient_ID', inplace=True)

# Save as a new file so we don't overwrite the old one!
output_path = os.path.join(output_dir, "physionet_clinical_24h_clean.csv")
master_df.to_csv(output_path)
print(f"[+] DONE! Processed matrix saved to: {output_path}")
print(f"[+] Final Matrix Shape: {master_df.shape[0]} patients, {master_df.shape[1]} features.")