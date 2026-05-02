import os
import GEOparse
import warnings
warnings.filterwarnings("ignore")

print("[*] INITIATING RNA-SEQ METADATA HARVESTER...")

data_dir = "/workspace/data/raw/rna-seq"
output_file = os.path.join(data_dir, "rnaseq_metadata_report.txt")

files = [f for f in os.listdir(data_dir) if f.endswith('.soft.gz')]

with open(output_file, 'w') as out_f:
    for file in files:
        file_path = os.path.join(data_dir, file)
        gse_id = file.split('_')[0]
        
        print(f"[+] Deep Scanning {gse_id}...")
        try:
            gse = GEOparse.get_GEO(filepath=file_path, silent=True)
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
            
            out_f.write(f"========================================\n")
            out_f.write(f"COHORT: {gse_id} | TOTAL PATIENTS: {len(gse.gsms)}\n")
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
            print(f"    [!] Failed to read {gse_id}: {e}")

print(f"\n[*] HARVEST COMPLETE. Report saved to {output_file}")