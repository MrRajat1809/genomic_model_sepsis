import os
import pandas as pd
import xgboost as xgb
import shap
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("==================================================")
print("[*] INITIATING SHAP EXPLAINABLE AI (XAI) ENGINE...")
print("==================================================")

# Directories
base_dir = "/workspace"
data_dir_gen = os.path.join(base_dir, "data/processed/ml_tensors")
data_dir_clin = os.path.join(base_dir, "data/processed/clinical_tensors")
model_dir = os.path.join(base_dir, "outputs/models")
fig_dir = os.path.join(base_dir, "outputs/figures")
os.makedirs(fig_dir, exist_ok=True)

# 1. Load the Models
print("\n[+] Loading Models...")
genetic_model = xgb.XGBClassifier()
genetic_model.load_model(os.path.join(model_dir, "xgboost_baseline.json"))

clinical_model = xgb.XGBClassifier()
clinical_model.load_model(os.path.join(model_dir, "icu_doctor_baseline.json"))

# 2. Load a Representative Background Sample (500 patients)
print("[+] Loading Representative Patient Tensors...")
X_gen = pd.read_csv(os.path.join(data_dir_gen, "X_master.csv.gz"), compression='gzip', nrows=500)
X_clin = pd.read_csv(os.path.join(data_dir_clin, "clinical_master_raw.csv.gz"), compression='gzip', nrows=500)

# Drop the labels/cheats from clinical
X_clin = X_clin.drop(columns=['Patient_ID', 'Sepsis_Outcome', 'ICU_Length_of_Stay'], errors='ignore')

# 3. SHAP Analysis: The Geneticist
print("\n[*] Calculating SHAP values for The Geneticist (This takes a moment)...")
explainer_gen = shap.TreeExplainer(genetic_model)
shap_values_gen = explainer_gen.shap_values(X_gen)

print("    -> Generating Genetic SHAP Summary Plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(shap_values_gen, X_gen, show=False)
plt.title("SHAP Summary: Genetic Predictors of Sepsis Mortality")
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, "shap_genetic_summary.png"), dpi=300, bbox_inches='tight')
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
plt.savefig(os.path.join(fig_dir, "shap_clinical_summary.png"), dpi=300, bbox_inches='tight')
plt.close()

print("\n" + "="*50)
print("[*] XAI ANALYSIS COMPLETE")
print(f"[*] Saved Genetic SHAP Plot to: {fig_dir}/shap_genetic_summary.png")
print(f"[*] Saved Clinical SHAP Plot to: {fig_dir}/shap_clinical_summary.png")
print("="*50)