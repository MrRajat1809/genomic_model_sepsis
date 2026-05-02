import os
import gzip

print("[*] INITIATING ULTRA-LOW MEMORY MICROARRAY HARVESTER...")

data_dir = "/workspace/data/raw/microarray"
output_file = "/workspace/data/raw/microarray/microarray_metadata_report.txt"

# Get all downloaded soft.gz files
files = [f for f in os.listdir(data_dir) if f.endswith('.soft.gz')]

with open(output_file, 'w', encoding='utf-8') as out_f:
    for file in files:
        file_path = os.path.join(data_dir, file)
        gse_id = file.split('_')[0]
        
        print(f"[+] Streaming {gse_id}...")
        try:
            meta_dict = {}
            sample_count = 0
            
            # Open the gzip file directly and read line-by-line (NO loading into RAM)
            with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    # Count patients
                    if line.startswith('^Sample ='):
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
            
            # Write results
            out_f.write(f"========================================\n")
            out_f.write(f"COHORT: {gse_id} | TOTAL PATIENTS: {sample_count}\n")
            out_f.write(f"========================================\n")
            
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

print(f"\n[*] MICROARRAY HARVEST COMPLETE. Report saved to {output_file}")