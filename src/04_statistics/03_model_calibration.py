"""
03_model_calibration.py

Calibration Metrics Computation.
Reads the raw prediction probabilities from the external Vault validation,
calculates the Brier Score, and computes the uniform bin coordinates for the 
calibration curve, exporting the raw data for downstream visualization.
"""

import warnings
from pathlib import Path

import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
METRICS_DIR = BASE_DIR / "outputs" / "metrics"

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("Calculating clinical calibration metrics...")
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD RAW PREDICTIONS
    # ---------------------------------------------------------
    pred_path = METRICS_DIR / "vault_predictions.csv"
    if not pred_path.exists():
        print(f"Error: Predictions file missing: {pred_path.name}")
        return

    preds = pd.read_csv(pred_path)
    y_true = preds['y_true']
    y_prob = preds['y_prob']

    # ---------------------------------------------------------
    # 2. CALCULATE METRICS & BINS
    # ---------------------------------------------------------
    brier = brier_score_loss(y_true, y_prob)
    print(f"External Vault Brier Score: {brier:.4f}")

    # Generate coordinates for the calibration curve (10 uniform bins)
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_prob, n_bins=10, strategy='uniform'
    )

    # ---------------------------------------------------------
    # 3. EXPORT RAW COORDINATES
    # ---------------------------------------------------------
    calib_df = pd.DataFrame({
        'mean_predicted_value': mean_predicted_value,
        'fraction_of_positives': fraction_of_positives
    })

    out_path = METRICS_DIR / "vault_calibration_coords.csv"
    calib_df.to_csv(out_path, index=False)
    
    # Save Brier score for the plotting script to read
    pd.DataFrame([{'brier_score': brier}]).to_csv(METRICS_DIR / "vault_brier_score.csv", index=False)
    
    print(f"Calibration coordinates exported to: {out_path.name}")

if __name__ == "__main__":
    main()