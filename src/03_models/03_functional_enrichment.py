"""
03_functional_enrichment.py

Biological Pathway Validation Module.
Utilizes the Enrichr API (via gseapy) to map the robust DEG signature 
to known Gene Ontology (GO) biological processes and KEGG disease pathways. 
This establishes the mechanistic and pathophysiological relevance of the 
identified biomarker genes to the clinical sepsis phenotype prior to 
machine learning dimensionality reduction.
"""

import warnings
from pathlib import Path

import gseapy as gp
import pandas as pd

# Suppress warnings for clean execution logging
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
# Resolves to Multimodal_prediction_sepsis/
BASE_DIR = Path(__file__).resolve().parents[2]
DEG_DIR = BASE_DIR / "data" / "processed" / "deg_tensors"

INPUT_FILE = DEG_DIR / "deg_top_percentile_features.csv"
OUTPUT_FILE = DEG_DIR / "functional_enrichment_results.csv"

# Target ontologies for immunological and functional mapping
DATABASES = [
    'GO_Biological_Process_2021', 
    'GO_Cellular_Component_2021', 
    'GO_Molecular_Function_2021', 
    'KEGG_2021_Human'
]

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    """Main execution workflow."""
    print("[*] Initiating functional enrichment analysis (gseapy)...")

    # ---------------------------------------------------------
    # 1. LOAD ELITE BIOMARKER GENES
    # ---------------------------------------------------------
    if not INPUT_FILE.exists():
        print(f"[ERROR] Required input not found: {INPUT_FILE.name}")
        return

    print("    -> Loading statistically robust DEG signature...")
    elite_df = pd.read_csv(INPUT_FILE)
    gene_list = elite_df['Gene'].tolist()
    print(f"       - Extracted {len(gene_list)} target genes for pathway mapping.")

    # ---------------------------------------------------------
    # 2. QUERY ENRICHR API
    # ---------------------------------------------------------
    print(f"    -> Interfacing with Enrichr API...")
    for db in DATABASES:
        print(f"       - Querying database: {db}")
        
    try:
        enr = gp.enrichr(
            gene_list=gene_list,
            gene_sets=DATABASES,
            organism='human',
            outdir=None,  # Suppress auto-plotting; custom plotting is handled separately
            cutoff=0.05   
        )
    except Exception as e:
        print(f"    [ERROR] Connection or API exception occurred: {e}")
        return

    results_df = enr.results
    if results_df.empty:
        print("    [!] Warning: No statistically significant pathways identified.")
        return

    # ---------------------------------------------------------
    # 3. FORMAT AND EXPORT RESULTS
    # ---------------------------------------------------------
    print("\n    -> Formatting and exporting enrichment profiles...")
    
    # Sort by Adjusted P-value for rigorous biological reporting
    if 'Adjusted P-value' in results_df.columns:
        results_df = results_df.sort_values(by='Adjusted P-value', ascending=True)

    results_df.to_csv(OUTPUT_FILE, index=False)
    print(f"       [+] Biological validation complete. Saved to: {OUTPUT_FILE.name}")
    
    print("\n" + "=" * 65)
    print("[*] Pathway mapping successfully executed.")
    print("=" * 65)

# ==========================================
# EXECUTION GUARD
# ==========================================
if __name__ == "__main__":
    main()