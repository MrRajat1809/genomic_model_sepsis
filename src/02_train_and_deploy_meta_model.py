"""
02_train_and_deploy_meta_model.py

Multimodal Meta-Model (Late Fusion Prototype).
Demonstrates a late-fusion architecture by training a Logistic Regression 
"Chief of Medicine" meta-learner on the probability outputs of the base 
genomic (Molecular Geneticist) and clinical (ICU Doctor) models. 
Avoids the dimensionality curse of early fusion and simulates a live, 
fused diagnostic dashboard.
"""

import warnings
from pathlib import Path

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib

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
    print("[*] BUILDING MULTIMODAL META-MODEL (LATE FUSION)")
    print("==================================================")

    # Ensure output directory exists
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the Base AI Brains
    print("\n[+] Loading Base Models...")
    gen_model_path = MODEL_DIR / "molecular_geneticist_baseline.json"
    clin_model_path = MODEL_DIR / "icu_doctor_baseline.json"

    if not gen_model_path.exists() or not clin_model_path.exists():
        print("[!] Critical Error: Base models not found. Train baseline models first.")
        return

    genetic_model = xgb.XGBClassifier()
    genetic_model.load_model(gen_model_path)

    clinical_model = xgb.XGBClassifier()
    clinical_model.load_model(clin_model_path)

    # 2. Load the Master Data to Train the Meta-Model
    print("[+] Loading Master Data for Meta-Training...")
    x_gen_path = DATA_DIR_GEN / "X_master.csv.gz"
    y_gen_path = DATA_DIR_GEN / "y_master.csv"
    x_clin_path = DATA_DIR_CLIN / "clinical_master_raw.csv.gz"

    if not x_gen_path.exists() or not x_clin_path.exists():
        print("[!] Critical Error: Processed tensors not found.")
        return

    X_gen_master = pd.read_csv(x_gen_path, compression='gzip')
    y_master = pd.read_csv(y_gen_path)
    X_clin_master = pd.read_csv(x_clin_path, compression='gzip')

    print("[*] Performing Synthetic Label-Matched Alignment...")
    # Separate clinical database by outcomes to match our genetic patients
    clin_dead = X_clin_master[X_clin_master['Sepsis_Outcome'] == 1]
    clin_alive = X_clin_master[X_clin_master['Sepsis_Outcome'] == 0]

    # Build a list of clinical patients that perfectly matches the 0/1 distribution of the genetic patients
    aligned_clin_rows = []
    pos_sample = clin_dead.sample(n=y_master['Mortality'].sum(), replace=True, random_state=42)
    neg_sample = clin_alive.sample(n=len(y_master) - y_master['Mortality'].sum(), replace=True, random_state=42)

    # Preserve the exact row order of y_master so genetic profiles match clinical profiles
    pos_idx, neg_idx = 0, 0
    for outcome in y_master['Mortality']:
        if outcome == 1:
            aligned_clin_rows.append(pos_sample.iloc[pos_idx])
            pos_idx += 1
        else:
            aligned_clin_rows.append(neg_sample.iloc[neg_idx])
            neg_idx += 1

    # Finalize the perfectly aligned matrices
    X_clin_aligned_full = pd.DataFrame(aligned_clin_rows).reset_index(drop=True)
    X_clin_aligned = X_clin_aligned_full.drop(columns=['Patient_ID', 'Sepsis_Outcome', 'ICU_Length_of_Stay'], errors='ignore')

    # Split a holdout set specifically to train the "Chief of Medicine" (Meta-Model)
    X_g_train, X_g_meta, X_c_train, X_c_meta, y_train, y_meta = train_test_split(
        X_gen_master, X_clin_aligned, y_master['Mortality'], test_size=0.3, random_state=42
    )

    # 3. Generate Base Predictions (Level 1 Features)
    print("[*] Generating Level-1 Predictions (The Expert Opinions)...")
    # The Geneticist's opinion
    meta_features_gen = genetic_model.predict_proba(X_g_meta)[:, 1]
    # The ICU Doctor's opinion
    meta_features_clin = clinical_model.predict_proba(X_c_meta)[:, 1]

    # Combine them into a new feature matrix (N_patients x 2 features)
    X_meta = np.column_stack((meta_features_clin, meta_features_gen))

    # 4. Train the Meta-Model (The Chief of Medicine)
    print("[*] Training Logistic Regression Meta-Learner...")
    meta_model = LogisticRegression(class_weight='balanced', random_state=42)
    meta_model.fit(X_meta, y_meta)

    # Save the Meta-Model
    meta_model_path = MODEL_DIR / "meta_model_logreg.pkl"
    joblib.dump(meta_model, meta_model_path)
    print(f"[+] Meta-Model saved to: {meta_model_path.name}")

    # Display the weights the Logistic Regression gave to each model
    weights = meta_model.coef_[0]
    print(f"    -> Weight assigned to Clinical Vitals: {weights[0]:.4f}")
    print(f"    -> Weight assigned to Genetic Profile: {weights[1]:.4f}")

    # =====================================================================
    # PART 2: LIVE INFERENCE SIMULATION
    # =====================================================================
    print("\n==================================================")
    print("[*] INITIALIZING LIVE MULTIMODAL INFERENCE ENGINE")
    print("==================================================")

    # Find a known critical patient for the simulation
    critical_idx = y_master[y_master['Mortality'] == 1].index[0]
    patient_genetics = X_gen_master.iloc[[critical_idx]].copy() 
    patient_clinical = X_clin_aligned.iloc[[critical_idx]].copy()

    # 1. Base Model Inference
    prob_sepsis_onset = clinical_model.predict_proba(patient_clinical)[0][1]
    prob_mortality = genetic_model.predict_proba(patient_genetics)[0][1]

    # 2. Meta-Model Fusion
    # Feed the two probabilities into the trained Logistic Regression model
    fusion_input = np.array([[prob_sepsis_onset, prob_mortality]])
    final_fused_risk = meta_model.predict_proba(fusion_input)[0][1]

    # 3. The Dashboard
    print("\n" + "="*55)
    print("              [ ICU MULTIMODAL DASHBOARD ]               ")
    print("="*55)

    print(f"-> THE ICU DOCTOR (Clinical Risk):     {prob_sepsis_onset * 100:.1f}%")
    print(f"-> THE GENETICIST (Genomic Risk):      {prob_mortality * 100:.1f}%\n")
    print(f"======> META-MODEL FUSED RISK SCORE:   {final_fused_risk * 100:.1f}% <======")

    if final_fused_risk > 0.70:
        print("\n[!!!] CODE RED: CRITICAL INTERVENTION REQUIRED [!!!]")
        print("      The Meta-Model has detected fatal physiological and genetic collapse.")
        print("      ACTION: Immediate broad-spectrum IV antibiotics and ICU transfer.")
    elif final_fused_risk > 0.40:
        print("\n[!] WARNING: ELEVATED RISK [!]")
        print("      Discordance between genes and vitals. Meta-model flags moderate danger.")
        print("      ACTION: Close monitoring. Do not discharge.")
    else:
        print("\n[+] ALL CLEAR: LOW RISK [+]")
        print("      Meta-model confirms patient is stable.")
        print("      ACTION: Standard ward care.")
    print("="*55)

if __name__ == "__main__":
    main()