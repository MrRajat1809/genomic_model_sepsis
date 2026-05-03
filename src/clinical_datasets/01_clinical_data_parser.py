import os
import glob
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

print("[*] INITIATING CLINICAL ICU DATA PARSER...")

# Paths mapped to your Docker workspace
base_dir = "/workspace/data/raw/physionet_2019/physionet.org/files/challenge-2019/1.0.0/training"
out_dir = "/workspace/data/processed/clinical_tensors"
os.makedirs(out_dir, exist_ok=True)

# Find all patient files in both Set A and Set B
set_a_files = glob.glob(os.path.join(base_dir, "training_setA", "*.psv"))
set_b_files = glob.glob(os.path.join(base_dir, "training_setB", "*.psv"))
all_files = set_a_files + set_b_files

print(f"    -> Discovered {len(set_a_files)} patients in Set A.")
print(f"    -> Discovered {len(set_b_files)} patients in Set B.")
print(f"    -> Total ICU Patients to Process: {len(all_files)}")

# Define which vitals/labs we want to extract (excluding demographics and labels)
dynamic_cols = [
    'HR', 'O2Sat', 'Temp', 'SBP', 'MAP', 'DBP', 'Resp', 'EtCO2', 
    'BaseExcess', 'HCO3', 'FiO2', 'pH', 'PaCO2', 'SaO2', 'AST', 'BUN', 
    'Alkalinephos', 'Calcium', 'Chloride', 'Creatinine', 'Bilirubin_direct', 
    'Glucose', 'Lactate', 'Magnesium', 'Phosphate', 'Potassium', 'Bilirubin_total', 
    'TroponinI', 'Hct', 'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets'
]

patient_records = []

print("\n[*] EXTRACTING PATIENT BIOMARKERS (This will take a few minutes)...")
# Process a subset first if you want to test, or the whole thing
for i, file_path in enumerate(all_files):
    if i % 5000 == 0 and i > 0:
        print(f"    -> Processed {i} / {len(all_files)} patients...")
        
    try:
        # Read the pipe-delimited file
        df = pd.read_csv(file_path, sep='|')
        
        # Extract Patient ID from filename (e.g., 'p00101.psv' -> 'p00101')
        patient_id = os.path.basename(file_path).split('.')[0]
        
        # 1. Target Label: Did they ever get Sepsis?
        # SepsisLabel is 1 if they got it, 0 otherwise
        sepsis_outcome = int(df['SepsisLabel'].max())
        
        # 2. Static Demographics (Take the first row since Age/Gender don't change)
        age = df['Age'].iloc[0]
        gender = df['Gender'].iloc[0]
        icu_los = df['ICULOS'].max() # Total hours in ICU
        
        # Initialize a dictionary for this patient
        patient_data = {
            'Patient_ID': patient_id,
            'Age': age,
            'Gender': gender,
            'ICU_Length_of_Stay': icu_los,
            'Sepsis_Outcome': sepsis_outcome
        }
        
        # 3. Dynamic Clinical Features: Calculate Min, Max, and Mean for Vitals/Labs
        for col in dynamic_cols:
            if col in df.columns:
                # Drop NaNs for the calculation
                valid_data = df[col].dropna()
                
                if len(valid_data) > 0:
                    patient_data[f'{col}_mean'] = valid_data.mean()
                    patient_data[f'{col}_min'] = valid_data.min()
                    patient_data[f'{col}_max'] = valid_data.max()
                else:
                    # If the patient never had this lab/vital taken, fill with NaN (we will impute later)
                    patient_data[f'{col}_mean'] = np.nan
                    patient_data[f'{col}_min'] = np.nan
                    patient_data[f'{col}_max'] = np.nan
                    
        patient_records.append(patient_data)
        
    except Exception as e:
        print(f"    [!] Error processing {patient_id}: {e}")

# Compile the Master Clinical DataFrame
print("\n[*] COMPILING MASTER CLINICAL TENSOR...")
master_df = pd.DataFrame(patient_records)

# Save to disk
out_file = os.path.join(out_dir, "clinical_master_raw.csv.gz")
master_df.to_csv(out_file, index=False, compression='gzip')

print("="*50)
print("[*] CLINICAL EXTRACTION COMPLETE.")
print(f"[*] Total Patients Engineered: {len(master_df)}")
print(f"[*] Total Clinical Features: {master_df.shape[1]}")
print(f"[*] Sepsis Prevalence: {(master_df['Sepsis_Outcome'].mean() * 100):.1f}%")
print(f"[*] Saved to: {out_file}")
print("="*50)