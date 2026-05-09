"""
01_download_raw_data.py

Reads the master GEO cohort list, filters for studies with >100 samples, 
and automates the download of microarray raw data (.soft.gz) from NCBI FTP servers.
Flags non-microarray datasets (e.g., RNA-seq) for manual downloading.
"""

import urllib.request
from pathlib import Path
import pandas as pd
from tqdm import tqdm

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
# Dynamically resolve paths from the script's location (src/geo_datasets/)
BASE_DIR = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
MA_DIR = RAW_DATA_DIR / "microarray"
SEQ_DIR = RAW_DATA_DIR / "rna-seq"
CSV_PATH = RAW_DATA_DIR / "geo_sepsis_master_menu.csv"

# ==========================================
# HELPER CLASSES & FUNCTIONS
# ==========================================
class DownloadProgressBar(tqdm):
    """Provides a visual progress bar for urllib downloads."""
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url: str, file_path: Path, desc: str):
    """Downloads a file with a progress bar, skipping if it already exists."""
    if file_path.exists():
        print(f"    [+] {desc} already exists. Skipping.")
        return
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
            urllib.request.urlretrieve(url, file_path, reporthook=t.update_to)
    except Exception as e:
        print(f"    [!] Failed to download {desc}: {e}")

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    print("[*] INITIATING UNIFIED RAW DATA ACQUISITION...")

    # Ensure target directories exist
    MA_DIR.mkdir(parents=True, exist_ok=True)
    SEQ_DIR.mkdir(parents=True, exist_ok=True)

    # Validate Master Menu exists
    if not CSV_PATH.exists():
        print(f"[!] Critical Error: {CSV_PATH.name} not found. Please run the dragnet notebook first.")
        return

    # Load and prep the Master Menu
    df = pd.read_csv(CSV_PATH)
    df['Samples'] = pd.to_numeric(df['Samples'], errors='coerce')

    # Filter for Cohorts > 100 samples
    target_df = df[df['Samples'] > 100].sort_values(by='Samples', ascending=False)
    print(f"[+] Found {len(target_df)} target cohorts (>100 samples).\n")

    manual_targets = []

    # Route and Download
    for index, row in target_df.iterrows():
        gse = row['Accession']
        platform_type = str(row['Type']).lower()
        
        # Microarray -> Automate .soft.gz download
        if 'array' in platform_type:
            print(f"[Automating] {gse} | {row['Samples']} patients | Microarray")
            prefix = gse[:-3] + "nnn"
            url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse}/soft/{gse}_family.soft.gz"
            dest = MA_DIR / f"{gse}_family.soft.gz"
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

    # Generate Manual Action Report
    print("\n" + "="*65)
    print("🎯 MANUAL EXTRACTION HITLIST (RNA-SEQ & COMPLEX)")
    print("="*65)
    if manual_targets:
        print("Download the raw count matrices from 'Supplementary Files' at the links below.")
        print(f"Place all downloaded files inside: {SEQ_DIR}\n")
        for t in manual_targets:
            print(f"[{t['Accession']}] - {t['Samples']} patients ({t['Type']})")
            print(f"Link: {t['URL']}\n")
    else:
        print("All datasets were downloaded automatically!")
        
    print("[*] STEP 1 (DATA ACQUISITION) COMPLETE.")

if __name__ == "__main__":
    main()