"""
03_rnaseq_harvester.py

A metadata parser for RNA-Seq GEO datasets.
Utilizes GEOparse to load .soft.gz files, extracting patient characteristics 
and clinical variables to help manually identify relevant cohorts.
Outputs a summary text report.
"""

import warnings
from pathlib import Path
import GEOparse

# Suppress GEOparse warnings for cleaner terminal output
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
# Dynamically resolve paths from the script's location (src/geo_datasets/)
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "raw" / "rna-seq"
OUTPUT_FILE = DATA_DIR / "rnaseq_metadata_report.txt"

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    print("[*] INITIATING RNA-SEQ METADATA HARVESTER...")

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

    print(f"[*] Found {len(files)} datasets to process. Writing report to: {OUTPUT_FILE.name}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for file_path in files:
            gse_id = file_path.name.split('_')[0]
            print(f"[+] Deep Scanning {gse_id}...")
            
            try:
                # Load the dataset silently using GEOparse (casting Path to str)
                gse = GEOparse.get_GEO(filepath=str(file_path), silent=True)
                meta_dict = {}
                
                for gsm_name, gsm in gse.gsms.items():
                    chars = gsm.metadata.get('characteristics_ch1', [])
                    for char in chars:
                        if ':' in char:
                            key, val = char.split(':', 1)
                            key = key.strip()
                            val = val.strip()
                            
                            if key not in meta_dict:
                                meta_dict[key] = set()
                            meta_dict[key].add(val)
                
                # Write results to report
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
                print(f"    [!] Failed to read {gse_id}: {e}")

    print(f"\n[*] RNA-SEQ HARVEST COMPLETE. Report saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()