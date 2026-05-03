import os
import pandas as pd
import numpy as np
import xgboost as xgb
import warnings
warnings.filterwarnings("ignore")

print("==================================================")
print("[*] INITIALIZING MULTIMODAL ICU SURVIVAL SYSTEM...")
print("==================================================")

# Directories
base_dir = "/workspace"
data_dir_gen = os.path.join(base_dir, "data/processed/ml_tensors")
data_dir_clin = os.path.join(base_dir, "data/processed/clinical_tensors")
model_dir = os.path.join(base_dir, "outputs/models")

# 1. Load the AI Brains (The Saved XGBoost Models)
print("\n[+] Waking up The Geneticist (Genomics Model)...")
genetic_model = xgb.XGBClassifier()
genetic_model.load_model(os.path.join(model_dir, "xgboost_baseline.json"))

print("[+] Waking up The ICU Doctor (Clinical Model)...")
clinical_model = xgb.XGBClassifier()
clinical_model.load_model(os.path.join(model_dir, "icu_doctor_baseline.json"))

# 2. Find REAL Critical Patients
print("\n[*] FETCHING LIVE PATIENT DATA (Retrieving known critical patients)...")

# We only need to load the first 500 rows to find a critical patient, saving memory
X_gen_master = pd.read_csv(os.path.join(data_dir_gen, "X_master.csv.gz"), compression='gzip', nrows=500)
y_gen_master = pd.read_csv(os.path.join(data_dir_gen, "y_master.csv"), nrows=500)

X_clin_master = pd.read_csv(os.path.join(data_dir_clin, "clinical_master_raw.csv.gz"), compression='gzip', nrows=500)

# Find the exact row index of the first patient who actually DIED in the genetics data
critical_gen_idx = y_gen_master[y_gen_master['Mortality'] == 1].index[0]
patient_genetics = X_gen_master.iloc[[critical_gen_idx]].copy() 

# Find the exact row index of the first patient who actually GOT SEPSIS in the clinical data
critical_clin_idx = X_clin_master[X_clin_master['Sepsis_Outcome'] == 1].index[0]
patient_clinical = X_clin_master.iloc[[critical_clin_idx]].drop(columns=['Patient_ID', 'Sepsis_Outcome', 'ICU_Length_of_Stay'], errors='ignore').copy()

# 3. Multimodal Inference
print("\n[*] EXECUTING MULTIMODAL INFERENCE...")

# The ICU Doctor evaluates vitals for Sepsis Onset
prob_sepsis_onset = clinical_model.predict_proba(patient_clinical)[0][1]

# The Geneticist evaluates DNA for Mortality Risk
prob_mortality = genetic_model.predict_proba(patient_genetics)[0][1]

# 4. Fusion Logic (The Decision Matrix)
print("\n" + "="*50)
print("              [ ICU DASHBOARD ]               ")
print("="*50)

print(f"-> SEPSIS ONSET RISK (Next 6 Hours):   {prob_sepsis_onset * 100:.1f}%")
print(f"-> GENETIC MORTALITY RISK (If Sepsis): {prob_mortality * 100:.1f}%\n")

if prob_sepsis_onset > 0.60 and prob_mortality > 0.60:
    print("[!!!] CODE RED: CRITICAL INTERVENTION REQUIRED [!!!]")
    print("      Patient is entering Sepsis AND has a highly lethal genetic signature.")
    print("      ACTION: Immediate broad-spectrum IV antibiotics and ICU transfer.")
    
elif prob_sepsis_onset > 0.60 and prob_mortality <= 0.60:
    print("[!] WARNING: SEPSIS ALERT [!]")
    print("    Patient is entering Sepsis, but possesses a resilient genetic signature.")
    print("    ACTION: Begin standard Sepsis-3 protocol.")
    
elif prob_sepsis_onset <= 0.60 and prob_mortality > 0.60:
    print("[-] MONITORING: HIGH-RISK GENETICS [-]")
    print("    Patient vitals are stable, but they are genetically vulnerable.")
    print("    ACTION: Monitor vitals closely. Do not discharge.")
    
else:
    print("[+] ALL CLEAR: LOW RISK [+]")
    print("    Patient vitals are stable and genetics are standard.")
    print("    ACTION: Standard ward care.")
print("="*50)