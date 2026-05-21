"""
03_rnaseq_harvester.py (Scaled for Endpoint Harmonization)

A metadata parser for RNA-Seq GEO datasets.
Utilizes GEOparse to load .soft.gz files, extracting patient characteristics 
and clinical variables. Maps these variables to specific patient IDs (GSMs) 
and exports structured CSV files to enable patient-level endpoint 
harmonization (e.g., 28-day mortality).
"""

import warnings
import pandas as pd
from pathlib import Path
import GEOparse

# Suppress GEOparse warnings for cleaner terminal output
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
# Dynamically resolve paths from the script's location
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "raw" / "rna-seq"

# Directory for the structured CSVs (Shared with Microarray)
CSV_OUT_DIR = BASE_DIR / "data" / "raw" / "geo_metadata"
OUTPUT_REPORT = DATA_DIR / "rnaseq_metadata_report.txt"

# Ensure output directory exists
CSV_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    print("[*] INITIATING SCALED RNA-SEQ HARVESTER (CSV EXTRACTION)...")

    # Ensure the data directory exists
    if not DATA_DIR.exists():
        print(f"[!] Directory not found: {DATA_DIR}")
        print("[!] Please run the download script and manually add RNA-seq files first.")
        return

    # Get all downloaded soft.gz files
    files = [f for f in DATA_DIR.iterdir() if f.name.endswith('.soft.gz')]
    
    if not files:
        print(f"[!] No .soft.gz files found in {DATA_DIR}.")
        return

    print(f"[*] Found {len(files)} datasets. Generating CSVs in {CSV_OUT_DIR.name}...")

    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as out_f:
        for file_path in files:
            gse_id = file_path.name.split('_')[0]
            print(f"[+] Deep Scanning {gse_id}...")
            
            try:
                # Load the dataset silently using GEOparse
                gse = GEOparse.get_GEO(filepath=str(file_path), silent=True)
                
                samples_list = []
                meta_dict = {} # For the summary report
                
                for gsm_name, gsm in gse.gsms.items():
                    current_sample = {'Patient_ID': gsm_name}
                    chars = gsm.metadata.get('characteristics_ch1', [])
                    
                    for char in chars:
                        if ':' in char:
                            key, val = char.split(':', 1)
                            key = key.strip().lower() # Lowercase for easier auditing
                            val = val.strip()
                            
                            current_sample[key] = val
                            
                            if key not in meta_dict:
                                meta_dict[key] = set()
                            meta_dict[key].add(val)
                            
                    samples_list.append(current_sample)

                # ==========================================
                # EXPORT STRUCTURED CSV
                # ==========================================
                if samples_list:
                    df = pd.DataFrame(samples_list)
                    csv_path = CSV_OUT_DIR / f"{gse_id}_metadata.csv"
                    df.to_csv(csv_path, index=False)
                    print(f"    -> Saved {len(df)} patients to {csv_path.name}")
                
                # ==========================================
                # WRITE SUMMARY REPORT (Original Logic)
                # ==========================================
                out_f.write("========================================\n")
                out_f.write(f"COHORT: {gse_id} | TOTAL PATIENTS: {len(gse.gsms)}\n")
                out_f.write("========================================\n")
                
                for key, vals in meta_dict.items():
                    val_list = list(vals)
                    if len(val_list) > 8:
                        display_vals = " | ".join(val_list[:5]) + f" ...and {len(val_list)-5} more unique values."
                    else:
                        display_vals = " | ".join(val_list)
                    
                    out_f.write(f"  {key.ljust(30)} --> {display_vals}\n")
                out_f.write("\n\n")
                
            except Exception as e:
                print(f"    [!] Failed to process {gse_id}: {e}")

    print(f"\n[*] RNA-SEQ HARVEST COMPLETE. Metadata CSVs ready in: {CSV_OUT_DIR}")

if __name__ == "__main__":
    main()