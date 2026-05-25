"""
06b_brier_calibration.py

Generates Figure 7 Panel B: Model Calibration Curve.
Loads pre-computed calibration coordinates and raw probability distributions
to render a two-part reliability diagram (calibration curve and histogram).
"""

import warnings
from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
FIG_OUT = BASE_DIR / "outputs" / "figures"

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("Generating Figure 7 Panel B: Calibration Curve...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------
    coords_path = METRICS_DIR / "vault_calibration_coords.csv"
    preds_path = METRICS_DIR / "vault_predictions.csv"
    brier_path = METRICS_DIR / "vault_brier_score.csv"
    
    missing_files = [f.name for f in [coords_path, preds_path, brier_path] if not f.exists()]
    if missing_files:
        print(f"Error: Missing files: {', '.join(missing_files)}. Run compute script first.")
        return

    coords_df = pd.read_csv(coords_path)
    preds_df = pd.read_csv(preds_path)
    brier_score = pd.read_csv(brier_path)['brier_score'].iloc[0]

    vault_probs = preds_df['y_prob']
    mean_predicted_value = coords_df['mean_predicted_value']
    fraction_of_positives = coords_df['fraction_of_positives']

    # ---------------------------------------------------------
    # 2. SETUP CANVAS
    # ---------------------------------------------------------
    sns.set_theme(style="ticks")
    fig = plt.figure(figsize=(6.5, 7.5))
    
    gs = gridspec.GridSpec(4, 1, hspace=0.3)
    
    # =========================================================
    # TOP PANEL: CALIBRATION CURVE
    # =========================================================
    ax1 = plt.subplot(gs[:3, 0])
    
    ax1.plot([0, 1], [0, 1], linestyle='--', color='#888888', linewidth=1.5, label='Perfectly Calibrated', zorder=0)
    
    ax1.plot(
        mean_predicted_value, fraction_of_positives, 
        marker='o', markersize=7, markeredgecolor='white', markeredgewidth=1.2,
        color='#d62828', linewidth=2.5, zorder=3,
        label=f'Model Calibration (Brier = {brier_score:.3f})'
    )
    
    ax1.set_aspect('equal', 'box')
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    
    ax1.set_title('Model Reliability & Calibration', fontsize=12, color='#222222', pad=15)
    ax1.set_ylabel('Observed Fraction of Positives', fontsize=11, color='#111111')
    ax1.tick_params(axis='both', labelsize=10, colors='#222222')
    ax1.grid(True, linestyle=':', color='#dddddd', alpha=0.8, zorder=0)
    
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.spines['left'].set_linewidth(1.0)
    ax1.spines['bottom'].set_linewidth(1.0)
    ax1.spines['left'].set_color('#333333')
    ax1.spines['bottom'].set_color('#333333')
    
    ax1.legend(loc='lower right', frameon=True, edgecolor='#cccccc', framealpha=0.95, fontsize=10)

    # =========================================================
    # BOTTOM PANEL: PROBABILITY DISTRIBUTION HISTOGRAM
    # =========================================================
    ax2 = plt.subplot(gs[3, 0], sharex=ax1)
    
    ax2.hist(
        vault_probs, bins=20, range=(0, 1), 
        color='#d62828', alpha=0.6, edgecolor='white', linewidth=1.0
    )
    
    ax2.set_xlabel('Predicted Probability of Mortality', fontsize=11, color='#111111')
    ax2.set_ylabel('Patient Count', fontsize=10, color='#111111')
    ax2.tick_params(axis='both', labelsize=10, colors='#222222')
    ax2.grid(axis='y', linestyle=':', color='#dddddd', alpha=0.8, zorder=0)
    
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.0)
    ax2.spines['bottom'].set_linewidth(1.0)
    ax2.spines['left'].set_color('#333333')
    ax2.spines['bottom'].set_color('#333333')

    # ---------------------------------------------------------
    # 3. FINAL AESTHETICS & EXPORT
    # ---------------------------------------------------------
    out_path = FIG_OUT / "Fig7B.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Calibration plot saved to: {out_path.name}")

if __name__ == "__main__":
    main()