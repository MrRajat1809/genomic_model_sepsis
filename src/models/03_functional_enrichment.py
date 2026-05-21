"""
03_functional_enrichment.py

Python equivalent of 'clusterProfiler' & 'STRING' from Banerjee et al.
Uses gseapy (Enrichr API) to map our 63 Elite Biomarker Genes to known 
biological pathways (GO Biological Processes and KEGG Pathways) to prove 
their relevance to Sepsis pathophysiology.
Saves the raw enrichment data to CSV for downstream custom visualization.
"""

import warnings
from pathlib import Path
import pandas as pd
import gseapy as gp

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
DEG_DIR = BASE_DIR / "data" / "processed" / "deg_tensors"

# We want to know what biological processes and disease pathways our genes trigger
DATABASES = [
    'GO_Biological_Process_2021', 
    'GO_Cellular_Component_2021', 
    'GO_Molecular_Function_2021', 
    'KEGG_2021_Human'
]

def main():
    print("[*] INITIATING FUNCTIONAL ENRICHMENT ANALYSIS (gseapy)...")

    # 1. Load the 63 Elite Genes
    elite_df = pd.read_csv(DEG_DIR / "Gold_Standard_Elite_DEGs.csv")
    gene_list = elite_df['Gene'].tolist()
    total_genes = len(gene_list)
    print(f"    -> Loaded {total_genes} Elite Biomarker Genes.")

    # 2. Query the Enrichr API
    print(f"    -> Querying Enrichr Databases: {', '.join(DATABASES)}...")
    try:
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=DATABASES,
            organism='human',
            outdir=None, 
            cutoff=0.05   
        )
    except Exception as e:
        print(f"[!] Error querying Enrichr API. Check internet connection. Details: {e}")
        return

    results_df = enr.results
    if results_df.empty:
        print("[!] No significant pathways found. This is highly unusual for 63 DEGs.")
        return

    # 3. Save detailed text results
    out_path = DEG_DIR / "Functional_Enrichment_Results.csv"
    results_df.to_csv(out_path, index=False)

    print(f"[*] SUCCESS! Biological validation complete. Raw enrichment data saved to {out_path.name}")

if __name__ == "__main__":
    main()