import os
import pandas as pd
import urllib.request
from tqdm import tqdm

print("[*] INITIATING UNIFIED RAW DATA ACQUISITION...")

# 1. Define Architecture
base_dir = "/workspace/data/raw"
ma_dir = os.path.join(base_dir, "microarray")
seq_dir = os.path.join(base_dir, "rna-seq")
csv_path = os.path.join(base_dir, "geo_sepsis_master_menu.csv")

os.makedirs(ma_dir, exist_ok=True)
os.makedirs(seq_dir, exist_ok=True)

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url, file_path, desc):
    if os.path.exists(file_path):
        print(f"    [+] {desc} already exists. Skipping.")
        return
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
            urllib.request.urlretrieve(url, file_path, reporthook=t.update_to)
    except Exception as e:
        print(f"    [!] Failed to download {desc}: {e}")

# 2. Load the Master Menu
if not os.path.exists(csv_path):
    print(f"[!] Critical Error: {csv_path} not found. Please run the dragnet notebook first.")
    exit()

df = pd.read_csv(csv_path)
df['Samples'] = pd.to_numeric(df['Samples'], errors='coerce')

# Filter for Cohorts > 100 samples
target_df = df[df['Samples'] > 100].sort_values(by='Samples', ascending=False)
print(f"[+] Found {len(target_df)} target cohorts (>100 samples).\n")

manual_targets = []

# 3. Route and Download
for index, row in target_df.iterrows():
    gse = row['Accession']
    platform_type = str(row['Type']).lower()
    
    # Microarray -> Automate .soft.gz download
    if 'array' in platform_type:
        print(f"[Automating] {gse} | {row['Samples']} patients | Microarray")
        prefix = gse[:-3] + "nnn"
        url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse}/soft/{gse}_family.soft.gz"
        dest = os.path.join(ma_dir, f"{gse}_family.soft.gz")
        download_file(url, dest, gse)
        
    # RNA-Seq / Other -> Flag for manual download
    else:
        print(f"[Manual]     {gse} | {row['Samples']} patients | {row['Type']}")
        manual_targets.append({
            'Accession': gse,
            'Samples': row['Samples'],
            'Type': row['Type'],
            'URL': f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse}"
        })

# 4. Generate Manual Action Report
print("\n" + "="*65)
print("🎯 MANUAL EXTRACTION HITLIST (RNA-SEQ & COMPLEX)")
print("="*65)
if manual_targets:
    print("Download the raw count matrices from 'Supplementary Files' at the links below.")
    print("Place all downloaded files inside: /workspace/data/raw/rna-seq/\n")
    for t in manual_targets:
        print(f"[{t['Accession']}] - {t['Samples']} patients ({t['Type']})")
        print(f"Link: {t['URL']}\n")
else:
    print("All datasets were downloaded automatically!")
    
print("[*] STEP 1 (DATA ACQUISITION) COMPLETE.")