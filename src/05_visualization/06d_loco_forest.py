"""
06d_loco_forest.py

Generates Figure 7 Panel D: LOCO Forest Plot.
Loads the pre-computed LOCO metrics and meta-analysis statistics to render
a standard meta-analytical forest plot.
"""

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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
    print("Generating Figure 7 Panel D: LOCO Forest Plot...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------
    forest_path = METRICS_DIR / "loco_forest_metrics.csv"
    meta_path = METRICS_DIR / "loco_meta_stats.csv"
    
    if not forest_path.exists() or not meta_path.exists():
        print("Error: Missing LOCO metric files. Run compute script first.")
        return

    df_forest = pd.read_csv(forest_path)
    meta_stats = pd.read_csv(meta_path).iloc[0]

    df_forest = df_forest.sort_values(by='AUC', ascending=True).reset_index(drop=True)

    # ---------------------------------------------------------
    # 2. SETUP CANVAS
    # ---------------------------------------------------------
    sns.set_theme(style="ticks")
    
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    y_pos = np.arange(len(df_forest))

    # ---------------------------------------------------------
    # 3. RENDER FOREST PLOT ELEMENTS
    # ---------------------------------------------------------
    # Error Bars
    ax.errorbar(
        df_forest['AUC'], y_pos, 
        xerr=[df_forest['AUC'] - df_forest['AUC_Lower'], df_forest['AUC_Upper'] - df_forest['AUC']],
        fmt='none', ecolor='#aaaaaa', elinewidth=1.5, capsize=4, zorder=1
    )

    # Individual Cohort Markers
    ax.scatter(
        df_forest['AUC'], y_pos, 
        color='#005b96', s=110, 
        edgecolor='white', linewidth=1.5, zorder=2
    )

    # Y-axis Labels
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{row['Cohort']} (n={int(row['N_Patients'])})" for _, row in df_forest.iterrows()], fontsize=10)

    # Text Annotations (AUC and 95% CI)
    for i, row in df_forest.iterrows():
        text_label = f"{row['AUC']:.2f} [{row['AUC_Lower']:.2f}-{row['AUC_Upper']:.2f}]"
        ax.text(
            1.02, i, text_label, 
            transform=ax.get_yaxis_transform(), 
            va='center', fontsize=10, color='#222222'
        )

    # Pooled Estimate Marker & Span
    pooled_auc = meta_stats['Pooled_AUC']
    pooled_lower = meta_stats['Pooled_Lower']
    pooled_upper = meta_stats['Pooled_Upper']
    I2 = meta_stats['I2']

    ax.axvline(
        x=pooled_auc, color='#d62828', linestyle='--', 
        linewidth=1.5, zorder=1, label=f'Pooled AUC: {pooled_auc:.2f}'
    )
    ax.axvspan(pooled_lower, pooled_upper, color='#d62828', alpha=0.15, zorder=0)

    # ---------------------------------------------------------
    # 4. FORMATTING & EXPORT
    # ---------------------------------------------------------
    ax.set_xlim([0.45, 1.0])
    ax.set_xlabel('Area Under the ROC Curve (AUROC)', fontsize=11, labelpad=10, color='#111111')
    ax.set_title(f"Leave-One-Cohort-Out Meta-Analysis (Heterogeneity $I^2$: {I2:.1f}%)", fontsize=12, pad=15, loc='left', color='#222222')
    
    ax.tick_params(axis='x', labelsize=10, colors='#222222')
    ax.tick_params(axis='y', length=0, labelsize=10, colors='#222222')
    
    ax.grid(axis='x', linestyle=':', color='#dddddd', alpha=0.8, zorder=0)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.spines['bottom'].set_color('#333333')
    ax.spines['bottom'].set_linewidth(1.0)
    
    ax.legend(
        loc='lower right', 
        frameon=True, 
        facecolor='white',
        edgecolor='#cccccc', 
        framealpha=0.95, 
        fontsize=10
    )
    
    plt.tight_layout()
    
    out_path = FIG_OUT / "Fig7D.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Forest plot saved to: {out_path.name}")

if __name__ == "__main__":
    main()