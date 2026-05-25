"""
03c_targeted_go_dotplot.py

Generates Panel C: Targeted GO Pathways Dot Plot.
Extracts the top global GO pathways and filters for targeted 
immunological mechanisms using predefined keywords.
Features dynamically scaled faceting for compact visualization.
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

GLOBAL_TOP_N = 20

ONTOLOGY_MAP = {
    'GO_Biological_Process_2021': 'Biological Process (BP)',
    'GO_Cellular_Component_2021': 'Cellular Component (CC)',
    'GO_Molecular_Function_2021': 'Molecular Function (MF)'
}

SEPSIS_KEYWORDS = [
    "neutrophil", "toll-like", "lipopolysaccharide", "interleukin", 
    "cytokine", "immune", "bacterial", "chemokine", "granule", 
    "nf-kappa", "t cell", "macrophage", "defense", "inflammatory"
]

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def clean_and_wrap_go_term(term):
    clean_text = term.split(' (GO:')[0].capitalize()
    return textwrap.fill(clean_text, width=38)

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("[*] Generating Panel C: Targeted GO Dot Plot...")
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
    print(f"    -> Extracting top {GLOBAL_TOP_N} global GO pathways...")
    df = df[df['Gene_set'].isin(ONTOLOGY_MAP.keys())].copy()
    df = df.sort_values(by='Adjusted P-value').head(GLOBAL_TOP_N).copy()

    df['Term_Clean'] = df['Term'].apply(clean_and_wrap_go_term)
    mask = df['Term_Clean'].str.lower().apply(lambda x: any(kw in x for kw in SEPSIS_KEYWORDS))
    plot_df = df[mask].copy()

    if plot_df.empty:
        print(f"[!] None of the Top {GLOBAL_TOP_N} pathways matched the targeted keywords.")
        return

    print(f"       - Retained {len(plot_df)} targeted pathways.")

    plot_df['Ontology'] = plot_df['Gene_set'].map(ONTOLOGY_MAP)
    plot_df['Count'] = plot_df['Overlap'].apply(lambda x: int(str(x).split('/')[0]))
    plot_df['GeneRatio'] = plot_df['Count'] / total_opt_genes
    plot_df = plot_df.sort_values(by=['Ontology', 'Adjusted P-value'], ascending=[True, False])

    # ---------------------------------------------------------
    # 2. DYNAMIC GEOMETRY CONFIGURATION
    # ---------------------------------------------------------
    print("    -> Configuring compact panel geometry...")
    active_ontologies = plot_df['Ontology'].unique()
    num_panels = len(active_ontologies)
    
    counts = [len(plot_df[plot_df['Ontology'] == o]) for o in active_ontologies]
    height_ratios = [max(1, c) for c in counts]

    sns.set_theme(style="whitegrid", rc={"axes.edgecolor": "#333333", "grid.alpha": 0.4})
    
    fig_height = max(4.0, sum(counts) * 0.4 + 2.5) 
    
    fig, axes = plt.subplots(
        num_panels, 1, figsize=(9, fig_height), sharex=True, 
        gridspec_kw={'hspace': 0.02, 'height_ratios': height_ratios}
    )
    
    if num_panels == 1:
        axes = [axes]
    
    cmap = "Spectral_r"
    norm = Normalize(vmin=plot_df['Adjusted P-value'].min(), vmax=plot_df['Adjusted P-value'].max())
    
    # ---------------------------------------------------------
    # 3. RENDER SCATTER PLOTS
    # ---------------------------------------------------------
    print("    -> Rendering dot plots...")
    for ax, ontology in zip(axes, active_ontologies):
        subset = plot_df[plot_df['Ontology'] == ontology]
            
        sns.scatterplot(
            data=subset, x='GeneRatio', y='Term_Clean', size='Count', 
            hue='Adjusted P-value', hue_norm=norm, sizes=(100, 350),     
            palette=cmap, edgecolor='black', linewidth=0.8, ax=ax, zorder=3
        )
        
        # Standardize axis limits to prevent variable padding
        ymin, ymax = ax.get_ylim()
        if ymin > ymax:  
            ax.set_ylim(len(subset) - 0.5, -0.5)
        else:            
            ax.set_ylim(-0.5, len(subset) - 0.5)

        ax.set_ylabel('')
        ax.annotate(ontology, xy=(1.02, 0.5), xycoords='axes fraction', rotation=270, ha='left', va='center', fontsize=11)
        
        if ax.legend_: 
            ax.legend_.remove()

    axes[-1].set_xlabel('Gene Ratio', fontsize=12, labelpad=12, fontweight='bold')
    
    # ---------------------------------------------------------
    # 4. GENERATE LEGENDS
    # ---------------------------------------------------------
    handles, labels = axes[0].get_legend_handles_labels()
            
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
            bbox_to_anchor=(0.35, 0.02), ncol=len(size_labels), frameon=False, title_fontsize=11
        )
    
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.60, 0.01, 0.25, 0.015]) 
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Adjusted P-value', fontsize=11, labelpad=8)
    cbar.formatter = ticker.ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((-3, 3))
    cbar.update_ticks()
    
    # ---------------------------------------------------------
    # 5. EXPORT
    # ---------------------------------------------------------
    out_path = FIG_OUT / "Fig4C.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[*] SUCCESS! Panel C saved to: {out_path.name}")

if __name__ == "__main__":
    main()