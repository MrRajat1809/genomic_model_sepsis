"""
06a_plot_combined_roc.py

Generates a unified 1x2 Validation Figure.
Panel 1: 5-Fold Cross-Validated ROC Curve Analysis (Internal Validation).
Panel 2: External Vault Validation ROC Curve with Bootstrapped 95% CI.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve, roc_auc_score

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
METRICS_DIR = BASE_DIR / "outputs" / "metrics"
FEATURE_DIR = BASE_DIR / "outputs" / "features"
FIG_OUT = BASE_DIR / "outputs" / "figures"

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("Generating unified ROC validation figure...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD DATA METRICS
    # ---------------------------------------------------------
    cv_data_path = METRICS_DIR / "cv_fold_predictions.csv"
    vault_data_path = METRICS_DIR / "vault_predictions.csv"
    vault_bounds_path = METRICS_DIR / "vault_roc_bounds.csv"
    feature_path = FEATURE_DIR / "optimal_biomarker_panel.csv"
    
    required_files = [cv_data_path, vault_data_path, vault_bounds_path]
    missing_files = [f.name for f in required_files if not f.exists()]
    
    if missing_files:
        print(f"Error: Missing required files: {', '.join(missing_files)}")
        print("Ensure the deployment pipeline has been executed.")
        return

    cv_df = pd.read_csv(cv_data_path)
    vault_df = pd.read_csv(vault_data_path)
    bounds_df = pd.read_csv(vault_bounds_path)

    # ---------------------------------------------------------
    # 2. SETUP CANVAS
    # ---------------------------------------------------------
    sns.set_theme(style="ticks")
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # =========================================================
    # PANEL 1: INTERNAL CROSS-VALIDATION ROC (LEFT)
    # =========================================================
    print("Rendering Internal CV ROC...")
    ax1 = axes[0]
    
    fold_y_trues = [group['y_true'].values for _, group in cv_df.groupby('Fold')]
    fold_y_probs = [group['y_prob'].values for _, group in cv_df.groupby('Fold')]

    tprs, aucs = [], []
    mean_fpr = np.linspace(0, 1, 100)

    for y_true, y_prob in zip(fold_y_trues, fold_y_probs):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        ax1.plot(fpr, tpr, color='#1f77b4', alpha=0.15, linewidth=1.0)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    std_tpr = np.std(tprs, axis=0)
    
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

    ax1.fill_between(mean_fpr, tprs_lower, tprs_upper, color='#005b96', alpha=0.15, label=r'$\pm$ 1 Std. Dev.')
    ax1.plot(
        mean_fpr, mean_tpr, color='#005b96', linewidth=2.5, 
        label=f'Mean CV ROC (AUC = {mean_auc:.2f} $\pm$ {std_auc:.2f})'
    )
    ax1.plot([0, 1], [0, 1], linestyle='--', lw=1.5, color='#888888', zorder=0)

    ax1.set_aspect('equal', 'box')
    ax1.set_xlim([-0.02, 1.02])
    ax1.set_ylim([-0.02, 1.02])
    
    ax1.set_title("Internal CV ROC - Harmonized Cohorts", fontsize=12, pad=15, color='#222222')
    ax1.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, color='#111111')
    ax1.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, color='#111111')
    ax1.tick_params(axis='both', labelsize=10, colors='#222222')
    ax1.grid(True, linestyle=':', color='#dddddd', alpha=0.8, zorder=0)
    
    ax1.legend(loc="lower right", fontsize=10, frameon=True, edgecolor='#cccccc', framealpha=0.95)

    # =========================================================
    # PANEL 2: EXTERNAL VAULT VALIDATION ROC (RIGHT)
    # =========================================================
    print("Rendering External Vault ROC...")
    ax2 = axes[1]

    y_true_vault = vault_df['y_true']
    y_prob_vault = vault_df['y_prob']
    vault_auc_exact = roc_auc_score(y_true_vault, y_prob_vault)

    v_fpr = bounds_df['fpr']
    v_tpr_mean = bounds_df['tpr_mean']
    v_tpr_lower = bounds_df['tpr_lower']
    v_tpr_upper = bounds_df['tpr_upper']

    ax2.fill_between(v_fpr, v_tpr_lower, v_tpr_upper, color='#d62828', alpha=0.15, label='95% Bootstrap CI')
    ax2.plot(
        v_fpr, v_tpr_mean, color='#d62828', linewidth=2.5, 
        label=f'Vault ROC (AUC = {vault_auc_exact:.2f})'
    )
    ax2.plot([0, 1], [0, 1], color='#888888', linestyle='--', linewidth=1.5, zorder=0)

    ax2.set_aspect('equal', 'box')
    ax2.set_xlim([-0.02, 1.02])
    ax2.set_ylim([-0.02, 1.02])
    
    ax2.set_title("External Generalization - Unharmonized Vault", fontsize=12, pad=15, color='#222222')
    ax2.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=11, color='#111111')
    ax2.set_ylabel("True Positive Rate (Sensitivity)", fontsize=11, color='#111111')
    ax2.tick_params(axis='both', labelsize=10, colors='#222222')
    ax2.grid(True, linestyle=':', color='#dddddd', alpha=0.8, zorder=0)
    
    ax2.legend(loc="lower right", fontsize=10, frameon=True, edgecolor='#cccccc', framealpha=0.95)

    # ---------------------------------------------------------
    # 3. FINAL AESTHETICS & EXPORT
    # ---------------------------------------------------------
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color('#333333')

    plt.tight_layout(w_pad=3.0)
    
    out_path = FIG_OUT / "Fig7A.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Validation figure saved to: {out_path.name}")

if __name__ == "__main__":
    main()