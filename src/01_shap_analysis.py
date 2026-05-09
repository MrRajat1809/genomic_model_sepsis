"""
01_shap_analysis.py

Explainable AI (XAI) Engine using SHAP (SHapley Additive exPlanations).
Calculates feature attributions for both the Molecular Geneticist (genomics) 
and ICU Doctor (clinical) models to determine the most impactful biomarkers 
driving mortality and sepsis onset predictions. Outputs global summary plots.
"""

import warnings
from pathlib import Path

import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt

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
FIG_DIR = BASE_DIR / "outputs" / "figures"

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    print("==================================================")
    print("[*] INITIATING SHAP EXPLAINABLE AI (XAI) ENGINE...")
    print("==================================================")

    # Ensure output directory exists
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the Models
    print("\n[+] Loading Models...")
    
    gen_model_path = MODEL_DIR / "molecular_geneticist_baseline.json"
    clin_model_path = MODEL_DIR / "icu_doctor_baseline.json"
    
    if not gen_model_path.exists() or not clin_model_path.exists():
        print(f"[!] Critical Error: Trained models not found in {MODEL_DIR.name}.")
        print("[!] Please run the baseline training scripts first.")
        return

    genetic_model = xgb.XGBClassifier()
    genetic_model.load_model(gen_model_path)

    clinical_model = xgb.XGBClassifier()
    clinical_model.load_model(clin_model_path)

    # 2. Load a Representative Background Sample (500 patients)
    print("[+] Loading Representative Patient Tensors...")
    x_gen_path = DATA_DIR_GEN / "X_master.csv.gz"
    x_clin_path = DATA_DIR_CLIN / "clinical_master_raw.csv.gz"
    
    if not x_gen_path.exists() or not x_clin_path.exists():
        print("[!] Critical Error: Processed tensors not found. Run data parsers first.")
        return

    X_gen = pd.read_csv(x_gen_path, compression='gzip', nrows=500)
    X_clin = pd.read_csv(x_clin_path, compression='gzip', nrows=500)

    # Drop the labels/cheats from clinical
    X_clin = X_clin.drop(columns=['Patient_ID', 'Sepsis_Outcome', 'ICU_Length_of_Stay'], errors='ignore')

    # 3. SHAP Analysis: The Geneticist
    print("\n[*] Calculating SHAP values for The Molecular Geneticist (This takes a moment)...")
    explainer_gen = shap.TreeExplainer(genetic_model)
    shap_values_gen = explainer_gen.shap_values(X_gen)

    print("    -> Generating Genetic SHAP Summary Plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_gen, X_gen, show=False)
    plt.title("SHAP Summary: Genetic Predictors of Sepsis Mortality")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_genetic_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()

    # 4. SHAP Analysis: The ICU Doctor
    print("\n[*] Calculating SHAP values for The ICU Doctor...")
    explainer_clin = shap.TreeExplainer(clinical_model)
    shap_values_clin = explainer_clin.shap_values(X_clin)

    print("    -> Generating Clinical SHAP Summary Plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_clin, X_clin, show=False)
    plt.title("SHAP Summary: Clinical Predictors of Sepsis Onset")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_clinical_summary.png"), dpi=300, bbox_inches='tight')
    plt.close()

    print("\n" + "="*50)
    print("[*] XAI ANALYSIS COMPLETE")
    print(f"[*] Saved Genetic SHAP Plot to: {FIG_DIR.name}/shap_genetic_summary.png")
    print(f"[*] Saved Clinical SHAP Plot to: {FIG_DIR.name}/shap_clinical_summary.png")
    print("="*50)

if __name__ == "__main__":
    main()