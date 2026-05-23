"""
02_rnaseq_harvester.py

A metadata parser for RNA-Seq GEO datasets.
Utilizes GEOparse to load .soft.gz files, extracting patient characteristics 
and clinical variables. Maps these variables to specific patient IDs (GSMs) 
and exports structured CSV files to enable downstream patient-level clinical 
endpoint harmonization.
"""

import warnings
from pathlib import Path

import GEOparse
import pandas as pd

# Suppress GEOparse warnings to maintain clean terminal output
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
# Resolves to Multimodal_prediction_sepsis/
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "raw" / "rna-seq"
CSV_OUT_DIR = BASE_DIR / "data" / "raw" / "geo_metadata"
OUTPUT_REPORT = DATA_DIR / "rnaseq_metadata_report.txt"

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    """Main execution workflow."""
    print("[*] Initiating scaled RNA-Seq metadata harvester...")

    if not DATA_DIR.exists():
        print(f"[ERROR] Directory not found: {DATA_DIR}")
        print("[ERROR] Execute the download script and verify manual RNA-seq acquisitions.")
        return

    # Identify target files
    files = list(DATA_DIR.glob("*.soft.gz"))
    if not files:
        print(f"[!] No .soft.gz files found in {DATA_DIR}. Exiting.")
        return

    print(f"[*] Identified {len(files)} target datasets.")
    CSV_OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_REPORT, 'w', encoding='utf-8') as out_f:
        for file_path in files:
            gse_id = file_path.name.split('_')[0]
            print(f"    -> Processing cohort {gse_id}...")
            
            try:
                # Load the dataset silently via GEOparse
                gse = GEOparse.get_GEO(filepath=str(file_path), silent=True)
                
                samples_list = []
                meta_dict = {}  # Tracks unique values for the text report
                
                # Extract metadata per patient sample (GSM)
                for gsm_name, gsm in gse.gsms.items():
                    current_sample = {'Patient_ID': gsm_name}
                    chars = gsm.metadata.get('characteristics_ch1', [])
                    
                    for char in chars:
                        if ':' in char:
                            key, val = char.split(':', 1)
                            key = key.strip().lower()  # Standardize keys
                            val = val.strip()
                            
                            current_sample[key] = val
                            
                            # Aggregate unique values for the summary report
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
                    print(f"       [+] Extracted {len(df)} patient records to {csv_path.name}")
                
                # ==========================================
                # WRITE SUMMARY REPORT
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
                print(f"       [ERROR] Failed to process {gse_id}: {e}")

    print(f"\n[*] Harvest complete. Metadata CSVs written to: {CSV_OUT_DIR.name}/")
    print(f"[*] Summary report generated: {OUTPUT_REPORT.name}")

# ==========================================
# EXECUTION GUARD
# ==========================================
if __name__ == "__main__":
    main()