"""
00_multimodal_inference_engine.py

Proof-of-Concept: Multimodal ICU Survival System.
Loads both the 'Molecular Geneticist' (transcriptomic) and 'ICU Doctor' (clinical)
XGBoost models. Simulates a live clinical environment by fetching a known critical
patient's data and running a fused inference decision matrix to trigger
mock clinical alerts.
"""

import warnings
from pathlib import Path

import pandas as pd
import xgboost as xgb

# Suppress warnings for cleaner terminal output
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
# Dynamically resolve paths from the script's location (src/)
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR_GEN = BASE_DIR / "data" / "processed" / "ml_tensors"
DATA_DIR_CLIN = BASE_DIR / "data" / "processed" / "clinical_tensors"
MODEL_DIR = BASE_DIR / "outputs" / "models"

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    print("==================================================")
    print("[*] INITIALIZING MULTIMODAL ICU SURVIVAL SYSTEM...")
    print("==================================================")

    # 1. Load the AI Brains (The Saved XGBoost Models)
    print("\n[+] Waking up The Molecular Geneticist (Genomics Model)...")
    genetic_model = xgb.XGBClassifier()
    gen_model_path = MODEL_DIR / "molecular_geneticist_baseline.json"
    
    if not gen_model_path.exists():
        print(f"[!] Critical Error: {gen_model_path.name} not found. Run the genetic baseline script first.")
        return
    genetic_model.load_model(gen_model_path)

    print("[+] Waking up The ICU Doctor (Clinical Model)...")
    clinical_model = xgb.XGBClassifier()
    clin_model_path = MODEL_DIR / "icu_doctor_baseline.json"
    
    if not clin_model_path.exists():
        print(f"[!] Critical Error: {clin_model_path.name} not found. Run the clinical baseline script first.")
        return
    clinical_model.load_model(clin_model_path)

    # 2. Find REAL Critical Patients
    print("\n[*] FETCHING LIVE PATIENT DATA (Retrieving known critical patients)...")

    # We only need to load the first 500 rows to find a critical patient, saving memory
    X_gen_master = pd.read_csv(DATA_DIR_GEN / "X_master.csv.gz", compression='gzip', nrows=500)
    y_gen_master = pd.read_csv(DATA_DIR_GEN / "y_master.csv", nrows=500)

    X_clin_master = pd.read_csv(DATA_DIR_CLIN / "clinical_master_raw.csv.gz", compression='gzip', nrows=500)

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

if __name__ == "__main__":
    main()