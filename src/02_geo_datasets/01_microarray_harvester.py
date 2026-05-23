"""
01_microarray_harvester.py

A low memory streaming parser for GEO .soft.gz microarray files.
Reads raw data line-by-line to map metadata to specific patient IDs (^SAMPLE),
and exports a structured CSV file for each cohort. This structured output 
enables downstream patient-level clinical endpoint harmonization.
"""

import gzip
from pathlib import Path

import pandas as pd

# ==========================================
# CONFIGURATION
# ==========================================
# Resolves to Multimodal_prediction_sepsis/
BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data" / "raw" / "microarray"
CSV_OUT_DIR = BASE_DIR / "data" / "raw" / "geo_metadata"
OUTPUT_REPORT = DATA_DIR / "microarray_metadata_report.txt"

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    """Main execution workflow."""
    print("[*] Initiating scaled microarray metadata harvester...")

    if not DATA_DIR.exists():
        print(f"[ERROR] Directory not found: {DATA_DIR}")
        print("[ERROR] Execute the raw data acquisition script prior to parsing.")
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
                samples_list = []
                current_sample = {}
                current_sample_id = None
                meta_dict = {}  # Tracks unique values for the text report
                
                # Stream file line-by-line to minimize memory footprint
                with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        
                        # Phase 1: Identify new Patient ID
                        if line.upper().startswith('^SAMPLE ='):
                            if current_sample_id is not None:
                                samples_list.append(current_sample)
                                
                            current_sample_id = line.split('=')[1].strip()
                            current_sample = {'Patient_ID': current_sample_id}
                        
                        # Phase 2: Extract Patient Metadata
                        elif line.startswith('!Sample_characteristics_ch1 =') and current_sample_id is not None:
                            char_str = line.split('=', 1)[1].strip()
                            
                            if ':' in char_str:
                                key, val = char_str.split(':', 1)
                                key = key.strip().lower()  # Standardize keys
                                val = val.strip()
                                
                                current_sample[key] = val
                                
                                # Aggregate unique values for the summary report
                                if key not in meta_dict:
                                    meta_dict[key] = set()
                                meta_dict[key].add(val)
                                
                    # Append the final patient record
                    if current_sample_id is not None:
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
                print(f"       [ERROR] Failed to process {gse_id}: {e}")

    print(f"\n[*] Harvest complete. Metadata CSVs written to: {CSV_OUT_DIR.name}/")
    print(f"[*] Summary report generated: {OUTPUT_REPORT.name}")

# ==========================================
# EXECUTION GUARD
# ==========================================
if __name__ == "__main__":
    main()