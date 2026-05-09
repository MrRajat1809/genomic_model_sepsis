"""
02_microarray_harvester.py

An ultra-low memory streaming parser for GEO .soft.gz files.
Reads microarray datasets line-by-line without loading into RAM,
extracting unique metadata fields (e.g., mortality, age, gender) 
to help manually identify cohorts with relevant clinical outcomes.
Outputs a summary text report.
"""

import gzip
from pathlib import Path

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
# Dynamically resolve paths from the script's location (src/geo_datasets/)
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "raw" / "microarray"
OUTPUT_FILE = DATA_DIR / "microarray_metadata_report.txt"

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    print("[*] INITIATING ULTRA-LOW MEMORY MICROARRAY HARVESTER...")

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

    print(f"[*] Found {len(files)} datasets to process. Writing report to: {OUTPUT_FILE.name}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_f:
        for file_path in files:
            gse_id = file_path.name.split('_')[0]
            print(f"[+] Streaming {gse_id}...")
            
            try:
                meta_dict = {}
                sample_count = 0
                
                # Open the gzip file directly and read line-by-line (NO loading into RAM)
                with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                    for line in f:
                        # Count patients (Added .upper() to catch ^SAMPLE = formatting)
                        if line.upper().startswith('^SAMPLE ='):
                            sample_count += 1
                        
                        # Extract metadata
                        elif line.startswith('!Sample_characteristics_ch1 ='):
                            # String format: !Sample_characteristics_ch1 = key: value
                            char_str = line.split('=', 1)[1].strip()
                            
                            if ':' in char_str:
                                key, val = char_str.split(':', 1)
                                key = key.strip()
                                val = val.strip()
                                
                                if key not in meta_dict:
                                    meta_dict[key] = set()
                                meta_dict[key].add(val)
                
                # Write results to report
                out_f.write("========================================\n")
                out_f.write(f"COHORT: {gse_id} | TOTAL PATIENTS: {sample_count}\n")
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
                print(f"    [!] Failed to stream {gse_id}: {e}")

    print(f"\n[*] MICROARRAY HARVEST COMPLETE. Report saved to: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()