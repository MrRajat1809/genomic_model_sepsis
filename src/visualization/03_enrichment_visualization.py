"""
03_enrichment_visualization.py

Generates a publication-ready clusterProfiler-style dot plot from the 
saved functional enrichment CSV. Facets by BP, CC, and MF.
Adheres to strict scientific aesthetics: no bold text, no manual titles, 
and features a horizontal continuous colorbar with scientific notation.
"""

import warnings
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Normalize
import matplotlib.ticker as ticker

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
DEG_DIR = BASE_DIR / "data" / "processed" / "deg_tensors"
FIG_OUT = BASE_DIR / "outputs" / "figures"

TOTAL_DEGS = 63 
TOP_N = 10 

ONTOLOGY_MAP = {
    'GO_Biological_Process_2021': 'Biological Process (BP)',
    'GO_Cellular_Component_2021': 'Cellular Component (CC)',
    'GO_Molecular_Function_2021': 'Molecular Function (MF)'
}

def clean_go_term(term):
    """Strips the GO ID for a cleaner y-axis."""
    return term.split(' (GO:')[0].capitalize()

def main():
    print("[*] INITIATING POLISHED SCIENTIFIC VISUALIZATION...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    
    csv_path = DEG_DIR / "Functional_Enrichment_Results.csv"
    if not csv_path.exists():
        print(f"[!] File not found: {csv_path}")
        return

    # 1. Load and Process Data
    df = pd.read_csv(csv_path)
    df = df[df['Gene_set'].isin(ONTOLOGY_MAP.keys())].copy()
    
    if df.empty:
        print("[!] No GO terms found in the CSV.")
        return

    # 2. Engineer Plotting Metrics
    df['Ontology'] = df['Gene_set'].map(ONTOLOGY_MAP)
    df['Term_Clean'] = df['Term'].apply(clean_go_term)
    df['Count'] = df['Overlap'].apply(lambda x: int(str(x).split('/')[0]))
    df['GeneRatio'] = df['Count'] / TOTAL_DEGS

    plot_data = []
    for ontology in ONTOLOGY_MAP.values():
        sub_df = df[df['Ontology'] == ontology].nsmallest(TOP_N, 'Adjusted P-value')
        plot_data.append(sub_df)
    
    plot_df = pd.concat(plot_data)

    # 3. Apply Xena-Style Theme (theme_bw equivalent)
    print("    -> Generating clean, faceted dot plot...")
    sns.set_theme(style="whitegrid", rc={
        "axes.edgecolor": "#333333", 
        "xtick.bottom": True, 
        "ytick.left": True,
        "grid.color": "grey",
        "grid.alpha": 0.3,
        "font.weight": "normal",
        "axes.labelweight": "normal"
    })
    
    # Increased height to 14 and hspace to 0.15 for more vertical breathing room
    fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True, gridspec_kw={'hspace': 0.15})
    
    # Establish a universal color normalization for the true colorbar
    cmap = "Spectral_r"
    norm = Normalize(vmin=plot_df['Adjusted P-value'].min(), vmax=plot_df['Adjusted P-value'].max())
    
    for ax, ontology in zip(axes, ONTOLOGY_MAP.values()):
        subset = plot_df[plot_df['Ontology'] == ontology].copy()
        subset = subset.sort_values(by='Adjusted P-value', ascending=False)
        
        if subset.empty:
            ax.set_visible(False)
            continue
            
        sns.scatterplot(
            data=subset, 
            x='GeneRatio', 
            y='Term_Clean', 
            size='Count', 
            hue='Adjusted P-value',
            hue_norm=norm,
            sizes=(50, 450),     
            palette=cmap, 
            edgecolor='black',
            linewidth=0.5,
            alpha=0.9,
            ax=ax
        )
        
        # Clean formatting: only data
        ax.set_ylabel('')
        ax.grid(True, linestyle='-', alpha=0.4, axis='y')
        
        # Add the clean facet label to the right
        ax.annotate(
            ontology, 
            xy=(1.02, 0.5), 
            xycoords='axes fraction', 
            rotation=270, 
            ha='left', 
            va='center', 
            fontsize=11,
            color='#111111'
        )
        
        # Strip the default ugly seaborn legend from individual panels
        if ax.get_legend():
            ax.get_legend().remove()

    axes[-1].set_xlabel('Gene Ratio', fontsize=12, labelpad=12)
    
    # 4. Constructing the Horizontal Bottom Legends
    # Adjust layout to make room for bottom legends
    fig.subplots_adjust(bottom=0.12) # Slightly reduced bottom margin since figure is taller
    
    # Extract just the 'Size' elements from seaborn's hidden legends
    handles, labels = axes[0].get_legend_handles_labels()
    size_handles, size_labels = [], []
    capture = False
    
    for h, l in zip(handles, labels):
        if l == 'Adjusted P-value':
            capture = False
        if capture:
            # Handle Line2D objects (newer matplotlib/seaborn)
            if hasattr(h, 'set_markerfacecolor'):
                h.set_markerfacecolor('#808080')
                h.set_markeredgecolor('black')
            # Fallback for PathCollection (older versions)
            elif hasattr(h, 'set_facecolor'):
                h.set_facecolor('#808080')
                h.set_edgecolor('black')
                
            size_handles.append(h)
            size_labels.append(l)
        if l == 'Count':
            capture = True

    # Build the Discrete Gene Count Legend (Bottom Left)
    fig.legend(
        size_handles, size_labels, 
        title="Gene Count", 
        loc='upper center', 
        bbox_to_anchor=(0.35, 0.06), 
        ncol=len(size_labels), 
        frameon=False,
        title_fontsize=11
    )
    
    # Build the Continuous P-Value Colorbar (Bottom Right)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.60, 0.035, 0.25, 0.012]) # Adjusted vertical position for taller figure
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Adjusted P-value', fontsize=11, labelpad=8)
    
    # Apply scientific notation formatting to the colorbar
    cbar.formatter = ticker.ScalarFormatter(useMathText=True)
    cbar.formatter.set_powerlimits((-3, 3))
    cbar.update_ticks()
    cbar.ax.xaxis.set_ticks_position('top')
    
    # 5. Export at 300 DPI
    out_path = FIG_OUT / "Fig4_Functional_Enrichment_DotPlot_Polished.pdf"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[*] SUCCESS! Scientific plot saved to: {out_path.name}")

if __name__ == "__main__":
    main()