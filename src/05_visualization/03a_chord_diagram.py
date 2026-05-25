"""
03a_chord_diagram.py

Generates Panel A: Global GO Chord Diagram.
Maps the optimal biomarker genes to the top enriched biological pathways.
Highlights targeted immunological pathways based on predefined keywords.
"""

import textwrap
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from pycirclize import Circos

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
DEG_DIR = BASE_DIR / "data" / "processed" / "deg_tensors"
FEATURE_DIR = BASE_DIR / "outputs" / "features"
FIG_OUT = BASE_DIR / "outputs" / "figures"

MAX_PATHWAYS = 20  

# Keywords for targeted pathway highlighting
SEPSIS_KEYWORDS = [
    "neutrophil", "toll-like", "lipopolysaccharide", "interleukin", 
    "cytokine", "immune", "bacterial", "chemokine", "granule", 
    "nf-kappa", "t cell", "macrophage"
]

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    print("[*] Generating Panel A: GO Chord Diagram...")
    FIG_OUT.mkdir(parents=True, exist_ok=True)
    
    csv_path = DEG_DIR / "functional_enrichment_results.csv"
    
    if not csv_path.exists():
        print(f"[ERROR] Missing enrichment data at: {csv_path.name}")
        return

    # ---------------------------------------------------------
    # 1. LOAD AND FILTER DATA
    # ---------------------------------------------------------
    print(f"    -> Extracting top {MAX_PATHWAYS} global pathways...")
    df = pd.read_csv(csv_path)
    df = df.sort_values(by="Adjusted P-value").head(MAX_PATHWAYS).copy()
    
    # Clean and wrap pathway labels
    df['Term_Clean'] = df['Term'].apply(lambda x: x.split(' (GO:')[0].capitalize())
    df['Term_Clean'] = df['Term_Clean'].apply(lambda x: textwrap.fill(x, width=22))

    # ---------------------------------------------------------
    # 2. BUILD ADJACENCY MATRIX
    # ---------------------------------------------------------
    all_genes = set()
    for genes in df['Genes']:
        all_genes.update(genes.split(';'))
    all_genes = sorted(list(all_genes))
    
    print(f"       - Mapped {len(all_genes)} target biomarker genes.")
    
    matrix = pd.DataFrame(0, index=df['Term_Clean'], columns=all_genes)
    
    for _, row in df.iterrows():
        pathway = row['Term_Clean']
        genes = row['Genes'].split(';')
        for gene in genes:
            if gene in matrix.columns:
                matrix.loc[pathway, gene] = 1

    # ---------------------------------------------------------
    # 3. RENDER CIRCOS GEOMETRY
    # ---------------------------------------------------------
    print("    -> Rendering Circos geometry...")
    circos = Circos.initialize_from_matrix(
        matrix, 
        space=1.5,  
        cmap="Spectral",  
        label_kws=dict(size=7, orientation="vertical") 
    )
    
    fig = circos.plotfig()
    
    # ---------------------------------------------------------
    # 4. APPLY VISUAL FORMATTING
    # ---------------------------------------------------------
    print("    -> Applying targeted pathway highlighting...")
    ax = fig.axes[0]
    
    for txt in ax.texts:
        label_text = txt.get_text()
        flat_label = label_text.replace('\n', ' ').lower()
        
        # Format Gene Nodes
        if label_text in all_genes:
            txt.set_weight('bold')
            txt.set_color('#111111')
            txt.set_fontsize(8)  
            
        # Format Targeted Pathways
        elif any(kw in flat_label for kw in SEPSIS_KEYWORDS):
            txt.set_weight('bold')
            txt.set_color('#db4325') 
            
        # Format Background Pathways
        else:
            txt.set_color('#777777')
            txt.set_fontsize(6)  
    
    # ---------------------------------------------------------
    # 5. EXPORT
    # ---------------------------------------------------------
    out_path = FIG_OUT / "Fig4A.pdf"
    fig.savefig(out_path, dpi=300, bbox_inches='tight', transparent=True)
    plt.close(fig)
    
    print(f"[*] SUCCESS! Panel A saved to: {out_path.name}")

if __name__ == "__main__":
    main()