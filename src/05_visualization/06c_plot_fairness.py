"""
06c_plot_fairness.py

Generates Figure 7 Panel C: Algorithmic Fairness Subgroup Plot.
Loads the pre-computed demographic parity metrics and renders a horizontal bar chart.
Features an Equivalence Margin and cleanly separated typography to match 
manuscript formatting standards.
"""

import warnings
from pathlib import Path

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
    print("Generating Figure 7 Panel C: Fairness Plot...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # 1. LOAD DATA
    # ---------------------------------------------------------
    metrics_path = METRICS_DIR / "vault_fairness_metrics.csv"
    if not metrics_path.exists():
        print(f"Error: Missing metrics file: {metrics_path.name}. Run compute script first.")
        return

    df_results = pd.read_csv(metrics_path)
    
    baseline_row = df_results[df_results['Category'] == 'Baseline']
    if baseline_row.empty:
        print("Error: Baseline AUROC missing from metrics.")
        return
    
    baseline_auc = baseline_row.iloc[0]['AUROC']
    
    # Isolate subgroups for plotting
    df_subgroups = df_results[df_results['Category'] != 'Baseline'].copy()
    df_subgroups.reset_index(drop=True, inplace=True)

    # ---------------------------------------------------------
    # 2. SETUP CANVAS
    # ---------------------------------------------------------
    sns.set_theme(style="ticks")
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    
    # ---------------------------------------------------------
    # 3. RENDER EQUIVALENCE MARGIN & BASELINE
    # ---------------------------------------------------------
    ax.axvspan(
        baseline_auc - 0.02, baseline_auc + 0.02, 
        color='#e0e0e0', alpha=0.5, zorder=0, 
        label='Equivalence Margin (±0.02)'
    )
    
    ax.axvline(
        x=baseline_auc, color='#333333', linestyle='--', 
        linewidth=1.5, zorder=1, label=f'Global Baseline ({baseline_auc:.3f})'
    )

    # ---------------------------------------------------------
    # 4. RENDER BARS & ANNOTATIONS
    # ---------------------------------------------------------
    palette = {"Biological Sex": "#1f77b4", "Age Bracket": "#d62828"}
    
    sns.barplot(
        data=df_subgroups, 
        x='AUROC', 
        y='Subgroup', 
        hue='Category',
        palette=palette,
        dodge=False,
        edgecolor='white',
        linewidth=1.5,
        alpha=0.9,
        zorder=3,
        ax=ax
    )

    # Annotate AUROC and N-count
    for i, row in df_subgroups.iterrows():
        n_val = int(row['N'])
        auc_val = row['AUROC']
        
        ax.text(
            auc_val + 0.015, i, 
            f'{auc_val:.3f} (n={n_val})', 
            va='center', ha='left', color='#222222', fontsize=10
        )

    # ---------------------------------------------------------
    # 5. FORMATTING & EXPORT
    # ---------------------------------------------------------
    ax.set_xlim([0.0, 1.05])
    
    ax.set_title('Algorithmic Fairness: Demographic Parity', fontsize=12, pad=15, color='#222222', loc='left')
    ax.set_xlabel('Area Under the ROC Curve (AUROC)', fontsize=11, labelpad=10, color='#111111')
    ax.set_ylabel('')
    
    ax.tick_params(axis='both', labelsize=10, colors='#222222')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_color('#333333')
    ax.spines['bottom'].set_color('#333333')
    
    ax.legend(
        loc='upper center', bbox_to_anchor=(0.5, -0.22), 
        ncol=2, frameon=False, fontsize=10
    )
    
    plt.tight_layout()
    out_path = FIG_OUT / "Fig7C.pdf"
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Fairness plot saved to: {out_path.name}")

if __name__ == "__main__":
    main()