import os
import GEOparse
import pandas as pd

BASE_DIR = "/workspace"
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")

# Create processed folder
os.makedirs(PROCESSED_DIR, exist_ok=True)

def process_geo_dataset(gse_id):
    filepath = os.path.join(RAW_DIR, f"{gse_id}_family.soft.gz")
    print(f"\n[*] Loading {gse_id} into memory (this takes 1-2 minutes)...")
    
    # Load the SOFT file
    gse = GEOparse.get_GEO(filepath=filepath)
    
    # ---------------------------------------------------------
    # 1. EXTRACT CLINICAL METADATA
    # ---------------------------------------------------------
    clinical_df = gse.phenotype_data
    clinical_file = os.path.join(PROCESSED_DIR, f"{gse_id}_clinical.csv")
    clinical_df.to_csv(clinical_file)
    print(f"[+] Saved Clinical Data: {clinical_df.shape[0]} patients, {clinical_df.shape[1]} variables -> {clinical_file}")
    
    # Print a few columns so we can see how they named Age, Sex, Severity, etc.
    print("\n--- Top Clinical Columns ---")
    for col in clinical_df.columns[:10]:
        print(f" - {col}")
        
    # ---------------------------------------------------------
    # 2. EXTRACT GENE EXPRESSION MATRIX
    # ---------------------------------------------------------
    print(f"\n[*] Pivoting gene expression matrix for {gse_id}...")
    # This automatically grabs the expression 'VALUE' for every gene across all patients
    expr_df = gse.pivot_samples('VALUE')
    
    expr_file = os.path.join(PROCESSED_DIR, f"{gse_id}_expression.csv")
    expr_df.to_csv(expr_file)
    print(f"[+] Saved Expression Data: {expr_df.shape[0]} genes, {expr_df.shape[1]} patients -> {expr_file}")


if __name__ == "__main__":
    # Process the Pediatric Benchmark Dataset first
    process_geo_dataset("GSE66099")
    
    # We will hold off on GSE65682 (Adults) until we verify the first one worked perfectly