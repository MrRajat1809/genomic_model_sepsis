"""
01_shap_analysis.py

Explainable AI (XAI) Engine using SHAP (SHapley Additive exPlanations).
Calculates feature attributions for both the Molecular Geneticist (genomics) 
and ICU Doctor (clinical) models. 

Upgrades for Publication:
- Extracts SHAP values for a high-density sample (1,000 patients).
- Isolates the top 20 features for each modality based on Mean Absolute SHAP.
- Normalizes raw feature values for gradient coloring.
- Exports 'long-format' CSVs for publication-grade visualization in R (ggplot2).
"""

import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
import shap

# Suppress warnings for cleaner terminal output
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR_GEN = BASE_DIR / "data" / "processed" / "ml_tensors"
DATA_DIR_CLIN = BASE_DIR / "data" / "processed" / "clinical_tensors"
MODEL_DIR = BASE_DIR / "outputs" / "models"
# NEW: The R-Bridge Export Directory
PLOT_DATA_OUT = BASE_DIR / "outputs" / "plot_data"

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def extract_shap_for_r(X, shap_values, top_n=20):
    """
    Extracts the top N features by mean absolute SHAP value, normalizes their 
    raw values for coloring, and formats them into a long dataframe for ggplot2.
    """
    # 1. Calculate global importance (Mean Absolute SHAP)
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    
    # 2. Get indices of the top N features
    top_indices = np.argsort(mean_abs_shap)[::-1][:top_n]
    
    records = []
    
    # 3. Extract and scale data
    for idx in top_indices:
        feat_name = X.columns[idx]
        s_vals = shap_values[:, idx]
        f_vals = X.iloc[:, idx].values
        
        # Min-Max scale the feature values so R can map them from 0 (Low) to 1 (High)
        f_min, f_max = np.nanmin(f_vals), np.nanmax(f_vals)
        if f_max > f_min:
            f_vals_scaled = (f_vals - f_min) / (f_max - f_min)
        else:
            f_vals_scaled = np.zeros_like(f_vals)
            
        # Append to records
        for i in range(len(s_vals)):
            records.append({
                "Feature": feat_name,
                "SHAP_Value": s_vals[i],
                "Feature_Value_Scaled": f_vals_scaled[i]
            })
            
    return pd.DataFrame(records)

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    print("==================================================")
    print("[*] INITIATING SHAP XAI EXPORT ENGINE (R-BRIDGE)...")
    print("==================================================")

    # Ensure output directory exists
    PLOT_DATA_OUT.mkdir(parents=True, exist_ok=True)

    # 1. Load the Models
    print("\n[+] Loading Models...")
    gen_model_path = MODEL_DIR / "molecular_geneticist_baseline.json"
    clin_model_path = MODEL_DIR / "icu_doctor_baseline.json"
    
    if not gen_model_path.exists() or not clin_model_path.exists():
        print(f"[!] Critical Error: Trained models not found in {MODEL_DIR.name}.")
        return

    genetic_model = xgb.XGBClassifier()
    genetic_model.load_model(gen_model_path)

    clinical_model = xgb.XGBClassifier()
    clinical_model.load_model(clin_model_path)

    # 2. Load a High-Density Sample
    print("[+] Loading Patient Tensors (N=1000 for statistical density)...")
    x_gen_path = DATA_DIR_GEN / "X_master.csv.gz"
    x_clin_path = DATA_DIR_CLIN / "clinical_master_raw.csv.gz"
    
    # We increase nrows to 1000 for a richer, more continuous visual distribution
    X_gen = pd.read_csv(x_gen_path, compression='gzip', nrows=1000)
    X_clin = pd.read_csv(x_clin_path, compression='gzip', nrows=1000)

    # Drop non-feature columns from clinical
    X_clin = X_clin.drop(columns=['Patient_ID', 'Sepsis_Outcome', 'ICU_Length_of_Stay'], errors='ignore')

    # 3. SHAP Analysis: The Geneticist
    print("\n[*] Processing The Molecular Geneticist...")
    explainer_gen = shap.TreeExplainer(genetic_model)
    shap_values_gen = explainer_gen.shap_values(X_gen)
    
    print("    -> Structuring Genomic SHAP Data for R...")
    df_shap_gen = extract_shap_for_r(X_gen, shap_values_gen, top_n=20)
    gen_csv_path = PLOT_DATA_OUT / "shap_genomic_export.csv"
    df_shap_gen.to_csv(gen_csv_path, index=False)
    print(f"    -> Saved to: {gen_csv_path.name}")

    # 4. SHAP Analysis: The ICU Doctor
    print("\n[*] Processing The ICU Doctor...")
    explainer_clin = shap.TreeExplainer(clinical_model)
    shap_values_clin = explainer_clin.shap_values(X_clin)

    print("    -> Structuring Clinical SHAP Data for R...")
    df_shap_clin = extract_shap_for_r(X_clin, shap_values_clin, top_n=20)
    clin_csv_path = PLOT_DATA_OUT / "shap_clinical_export.csv"
    df_shap_clin.to_csv(clin_csv_path, index=False)
    print(f"    -> Saved to: {clin_csv_path.name}")

    print("\n" + "="*50)
    print("[*] SHAP EXPORT COMPLETE")
    print("[*] Ready for ggplot2/ggsci visualization.")
    print("="*50)

if __name__ == "__main__":
    main()