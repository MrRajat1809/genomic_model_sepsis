"""
02_microarray_harvester.py (Scaled for Endpoint Harmonization)

An ultra-low memory streaming parser for GEO .soft.gz files.
Reads microarray datasets line-by-line. Instead of just aggregating 
unique values, it now maps metadata to specific patient IDs (^SAMPLE) 
and exports a structured CSV file for each cohort. This enables 
precise, patient-level endpoint harmonization (e.g., 28-day mortality).
"""

import gzip
import pandas as pd
from pathlib import Path

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
# Dynamically resolve paths from the script's location
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "raw" / "microarray"

# New Directory for the structured CSVs
CSV_OUT_DIR = BASE_DIR / "data" / "raw" / "geo_metadata"
OUTPUT_REPORT = DATA_DIR / "microarray_metadata_report.txt"

# Ensure output directory exists
CSV_OUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    print("[*] INITIATING SCALED MICROARRAY HARVESTER (CSV EXTRACTION)...")

    # Ensure the data directory exists
    if not DATA_DIR.exists():
        print(f"[!] Directory not found: {DATA_DIR}")
        print("[!] Please run the download script first.")
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
            print(f"[+] Processing {gse_id}...")
            
            try:
                samples_list = []
                current_sample = {}
                current_sample_id = None
                meta_dict = {} # For the summary report
                
                # Open the gzip file directly and read line-by-line
                with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        
                        # 1. Catch the Patient ID
                        if line.upper().startswith('^SAMPLE ='):
                            # Save the previous patient before starting a new one
                            if current_sample_id is not None:
                                samples_list.append(current_sample)
                            
                            current_sample_id = line.split('=')[1].strip()
                            current_sample = {'Patient_ID': current_sample_id}
                        
                        # 2. Catch the Metadata fields for the current patient
                        elif line.startswith('!Sample_characteristics_ch1 =') and current_sample_id is not None:
                            char_str = line.split('=', 1)[1].strip()
                            
                            if ':' in char_str:
                                key, val = char_str.split(':', 1)
                                key = key.strip().lower() # Lowercase for easier auditing later
                                val = val.strip()
                                
                                current_sample[key] = val
                                
                                # Also track unique values for the text report
                                if key not in meta_dict:
                                    meta_dict[key] = set()
                                meta_dict[key].add(val)
                                
                # Append the very last patient in the file
                if current_sample_id is not None:
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
                out_f.write(f"COHORT: {gse_id} | TOTAL PATIENTS: {len(samples_list)}\n")
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

    print(f"\n[*] HARVEST COMPLETE. Metadata CSVs ready in: {CSV_OUT_DIR}")

if __name__ == "__main__":
    main()