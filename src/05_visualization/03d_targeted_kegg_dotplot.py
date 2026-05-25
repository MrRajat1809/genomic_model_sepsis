"""
03d_targeted_kegg_dotplot.py

Generates Panel D: Targeted KEGG Pathways Dot Plot.
Extracts the top global KEGG pathways and filters for targeted 
immunological mechanisms using predefined keywords.
Features dynamically scaled geometry for compact visualization.
"""

import textwrap
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from matplotlib.colors import Normalize

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
DEG_DIR = BASE_DIR / "data" / "processed" / "deg_tensors"
FEATURE_DIR = BASE_DIR / "outputs" / "features"
FIG_OUT = BASE_DIR / "outputs" / "figures"

GLOBAL_TOP_N = 15
TARGET_DB = 'KEGG_2021_Human'

# Keywords for targeted pathway highlighting
SEPSIS_KEYWORDS = [
    "neutrophil", "toll-like", "lipopolysaccharide", "interleukin", 
    "cytokine", "immune", "bacterial", "chemokine", "granule", 
    "nf-kappa", "t cell", "macrophage", "infection", "viral", "pathogen"
]

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def clean_and_wrap_kegg_term(term):
    clean_text = term.capitalize()
    return textwrap.fill(clean_text, width=38)

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("[*] Generating Panel D: Targeted KEGG Dot Plot...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    
    csv_path = DEG_DIR / "functional_enrichment_results.csv"
    opt_path = FEATURE_DIR / "optimal_biomarker_panel.csv"
    
    if not csv_path.exists() or not opt_path.exists():
        print("[ERROR] Required input files missing.")
        return
    
    total_opt_genes = len(pd.read_csv(opt_path))
    df = pd.read_csv(csv_path)
    
    # ---------------------------------------------------------
    # 1. LOAD AND FILTER DATA
    # ---------------------------------------------------------
    print(f"    -> Extracting top {GLOBAL_TOP_N} global KEGG pathways...")
    df = df[df['Gene_set'] == TARGET_DB].copy()
    df = df.sort_values(by='Adjusted P-value').head(GLOBAL_TOP_N).copy()

    df['Term_Clean'] = df['Term'].apply(clean_and_wrap_kegg_term)
    mask = df['Term_Clean'].str.lower().apply(lambda x: any(kw in x for kw in SEPSIS_KEYWORDS))
    plot_df = df[mask].copy()

    if plot_df.empty:
        print(f"[!] None of the Top {GLOBAL_TOP_N} KEGG pathways matched the targeted keywords.")
        return

    print(f"       - Retained {len(plot_df)} targeted KEGG pathways.")

    plot_df['Count'] = plot_df['Overlap'].apply(lambda x: int(str(x).split('/')[0]))
    plot_df['GeneRatio'] = plot_df['Count'] / total_opt_genes
    plot_df = plot_df.sort_values(by='Adjusted P-value', ascending=False)

    # ---------------------------------------------------------
    # 2. DYNAMIC GEOMETRY CONFIGURATION
    # ---------------------------------------------------------
    print("    -> Configuring compact panel geometry...")
    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "#333333", "grid.alpha": 0.4})
    
    fig_height = max(3.0, len(plot_df) * 0.4 + 2.0) 
    fig, ax = plt.subplots(figsize=(9, fig_height))
    
    cmap = "Spectral_r"
    norm = Normalize(vmin=plot_df['Adjusted P-value'].min(), vmax=plot_df['Adjusted P-value'].max())
    
    # ---------------------------------------------------------
    # 3. RENDER SCATTER PLOT
    # ---------------------------------------------------------
    print("    -> Rendering dot plot...")
    sns.scatterplot(
        data=plot_df, x='GeneRatio', y='Term_Clean', size='Count', 
        hue='Adjusted P-value', hue_norm=norm, sizes=(100, 350),     
        palette=cmap, edgecolor='black', linewidth=0.8, ax=ax, zorder=3
    )
    
    # Standardize axis limits for uniform padding
    ymin, ymax = ax.get_ylim()
    if ymin > ymax:  
        ax.set_ylim(len(plot_df) - 0.5, -0.5)
    else:            
        ax.set_ylim(-0.5, len(plot_df) - 0.5)
    
    ax.set_ylabel('')
    ax.annotate("KEGG Pathway", xy=(1.02, 0.5), xycoords='axes fraction', 
                rotation=270, ha='left', va='center', fontsize=11, fontweight='bold')
    
    if ax.legend_: 
        ax.legend_.remove()

    ax.set_xlabel('Gene Ratio', fontsize=12, labelpad=12, fontweight='bold')
    fig.subplots_adjust(bottom=0.3) 
    
    # ---------------------------------------------------------
    # 4. GENERATE LEGENDS
    # ---------------------------------------------------------
    handles, labels = ax.get_legend_handles_labels()
            
    size_handles, size_labels = [], []
    capture = False
    
    for h, l in zip(handles, labels):
        if l == 'Adjusted P-value': capture = False
        if capture:
            try:
                h.set_facecolor('#808080')
                h.set_edgecolor('black')
            except Exception: 
                pass
            size_handles.append(h)
            size_labels.append(l)
        if l == 'Count': capture = True

    if size_handles:
        fig.legend(
            size_handles, size_labels, title="Gene Count", loc='upper center', 
            bbox_to_anchor=(0.35, 0.05), ncol=len(size_labels), frameon=False, title_fontsize=11
        )
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.60, 0.05, 0.25, 0.02]) 
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Adjusted P-value', fontsize=11, labelpad=8)
    cbar.formatter = ticker.ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((-3, 3))
    cbar.update_ticks()
    
    # ---------------------------------------------------------
    # 5. EXPORT
    # ---------------------------------------------------------
    out_path = FIG_OUT / "Fig4D.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[*] SUCCESS! Panel D saved to: {out_path.name}")

if __name__ == "__main__":
    main()