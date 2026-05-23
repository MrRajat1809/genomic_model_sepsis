"""
03_model_calibration.py

Clinical Reliability and Calibration Analysis.
Evaluates the real-world clinical reliability of the optimized XGBoost model.
Calculates the Brier Score and generates a publication-ready Calibration Curve 
(Reliability Diagram) strictly on the independent holdout Vault (GSE65682).
"""

import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# Suppress warnings for clean execution logging
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
# Resolves to Multimodal_prediction_sepsis/
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "processed" / "ml_tensors"
FEATURE_DIR = BASE_DIR / "outputs" / "features"
MODEL_DIR = BASE_DIR / "outputs" / "models"
FIG_OUT = BASE_DIR / "outputs" / "figures"

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    """Main execution workflow."""
    print("[*] Initiating clinical reliability and calibration analysis...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD OPTIMAL FEATURES
    # ---------------------------------------------------------
    features_path = FEATURE_DIR / "optimal_biomarker_panel.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"[ERROR] Optimal features list not found at: {features_path.name}")
    
    optimal_genes = pd.read_csv(features_path)['Optimal_Genes'].tolist()
    num_genes = len(optimal_genes)
    print(f"    -> Loaded optimal biomarker signature: {num_genes} features.")

    # ---------------------------------------------------------
    # 2. LOAD QUARANTINED VAULT DATA
    # ---------------------------------------------------------
    print("    -> Loading quarantined holdout Vault tensor (GSE65682)...")
    X_vault_full = pd.read_csv(DATA_DIR / "X_vault.csv.gz", compression='gzip')
    y_vault_df = pd.read_csv(DATA_DIR / "y_vault.csv")
    
    # Subset to the optimal signature
    X_vault = X_vault_full[optimal_genes]
    
    target_col = 'Mortality' if 'Mortality' in y_vault_df.columns else y_vault_df.columns[0]
    y_vault = y_vault_df[target_col].astype(int)

    # ---------------------------------------------------------
    # 3. LOAD LOCKED CLINICAL MODEL
    # ---------------------------------------------------------
    model_name = f"sepsis_xgboost_{num_genes}genes_deploy_v1.json"
    model_path = MODEL_DIR / model_name
    
    if not model_path.exists():
        raise FileNotFoundError(f"[ERROR] Locked model weights not found: {model_name}")
        
    print(f"    -> Loading locked XGBoost architecture: {model_name}")
    model = xgb.XGBClassifier()
    model.load_model(model_path)

    # ---------------------------------------------------------
    # 4. PREDICT & CALCULATE METRICS
    # ---------------------------------------------------------
    print("    -> Calculating predictive probabilities on the un-harmonized Vault...")
    vault_probs = model.predict_proba(X_vault)[:, 1]

    # Calculate Calibration Metrics
    brier = brier_score_loss(y_vault, vault_probs)
    print("\n" + "-" * 65)
    print("CLINICAL RELIABILITY METRICS")
    print("-" * 65)
    print(f"    -> Brier Score: {brier:.4f} (Ideal = 0.00, Baseline = ~0.25)")
    
    # Generate coordinates for the calibration curve (10 uniform bins)
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_vault, vault_probs, n_bins=10, strategy='uniform'
    )

    # ---------------------------------------------------------
    # 5. GENERATE PUBLICATION FIGURE
    # ---------------------------------------------------------
    print("\n    -> Generating Reliability Diagram (Calibration Curve)...")
    
    fig = plt.figure(figsize=(9, 9))
    gs = gridspec.GridSpec(4, 1, hspace=0.4)
    
    # --- TOP PANEL: Calibration Curve ---
    ax1 = plt.subplot(gs[:3, 0])
    
    # Ideal perfectly calibrated line
    ax1.plot([0, 1], [0, 1], linestyle='--', color='#888888', linewidth=1.5, label='Perfectly Calibrated')
    
    # Model's empirical curve
    ax1.plot(
        mean_predicted_value, fraction_of_positives, marker='o', markersize=8,
        color='#db4325', linewidth=2.5, label=f'{num_genes}-Gene XGBoost (Brier = {brier:.3f})'
    )
    
    # Formatting Top Panel
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_color('#aaaaaa')
    ax1.spines['bottom'].set_color('#aaaaaa')
    ax1.tick_params(colors='#555555')
    
    ax1.set_ylabel('True Probability in Patients (Fraction of Positives)', fontsize=12, color='#444444', labelpad=10)
    ax1.set_title(f'Model Calibration & Reliability (Vault Cohort: GSE65682)', fontsize=14, color='#333333', pad=15, fontweight='bold')
    ax1.legend(loc='upper left', frameon=False, fontsize=11)
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_ylim([-0.05, 1.05])
    ax1.grid(True, linestyle=':', color='#f0f0f0')

    # --- BOTTOM PANEL: Histogram of Predictions ---
    ax2 = plt.subplot(gs[3, 0], sharex=ax1)
    
    ax2.hist(vault_probs, bins=20, range=(0, 1), color='#4a6fe3', alpha=0.7, edgecolor='white', linewidth=1.2)
    
    # Formatting Bottom Panel
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_color('#aaaaaa')
    ax2.spines['bottom'].set_color('#aaaaaa')
    ax2.tick_params(colors='#555555')
    
    ax2.set_xlabel('Predicted Probability of Mortality', fontsize=12, color='#444444', labelpad=10, fontweight='bold')
    ax2.set_ylabel('Patient Count', fontsize=11, color='#444444')
    ax2.grid(axis='y', linestyle=':', color='#f0f0f0')

    plt.tight_layout()
    
    pdf_out = FIG_OUT / f"Fig_Calibration_Curve_{num_genes}Genes.pdf"
    plt.savefig(pdf_out, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"[*] Analysis Complete. Plot saved to: {pdf_out.name}")

# ==========================================
# EXECUTION GUARD
# ==========================================
if __name__ == "__main__":
    main()