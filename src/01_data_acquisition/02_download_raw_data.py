"""
02_download_raw_data.py

Parses the master GEO cohort index, filters for studies containing >100 samples, 
and automates the download of microarray raw data (.soft.gz) from NCBI FTP servers.
Non-microarray datasets (e.g., RNA-seq) are flagged for manual acquisition.
"""

import urllib.request
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# ==========================================
# CONFIGURATION
# ==========================================
# Resolves to Multimodal_prediction_sepsis/
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
    
    def update_to(self, b: int = 1, bsize: int = 1, tsize: int = None):
        """Updates the progress bar state based on downloaded bytes."""
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_file(url: str, file_path: Path, desc: str) -> None:
    """
    Downloads a file via HTTP/FTP with a visual progress bar.
    Skips the download if the target file already exists.
    
    Args:
        url (str): The source URL of the file.
        file_path (Path): The local destination path.
        desc (str): Description/name for the progress bar.
    """
    if file_path.exists():
        print(f"    [INFO] {desc} already exists locally. Skipping download.")
        return
        
    try:
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
            urllib.request.urlretrieve(url, file_path, reporthook=t.update_to)
    except Exception as e:
        print(f"    [ERROR] Failed to download {desc}: {e}")

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    """Main execution workflow."""
    print("[*] Initiating unified raw data acquisition protocol...")

    # Ensure target directories exist
    MA_DIR.mkdir(parents=True, exist_ok=True)
    SEQ_DIR.mkdir(parents=True, exist_ok=True)

    # Validate Master Menu existence
    if not CSV_PATH.exists():
        print(f"[ERROR] {CSV_PATH.name} not found. Execute the dragnet script first.")
        return

    # Load and preprocess the Master Menu
    df = pd.read_csv(CSV_PATH)
    df['Samples'] = pd.to_numeric(df['Samples'], errors='coerce')

    # Filter for cohorts exceeding the 100-sample threshold
    target_df = df[df['Samples'] > 100].sort_values(by='Samples', ascending=False)
    print(f"[*] Identified {len(target_df)} target cohorts (>100 samples).\n")

    manual_targets = []

    # Execute routing and download logic
    for _, row in target_df.iterrows():
        gse = row['Accession']
        platform_type = str(row['Type']).lower()
        
        if 'array' in platform_type:
            print(f"[Automated] {gse} | N={row['Samples']} | Microarray")
            prefix = gse[:-3] + "nnn"
            url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse}/soft/{gse}_family.soft.gz"
            dest = MA_DIR / f"{gse}_family.soft.gz"
            download_file(url, dest, gse)
            
        else:
            print(f"[Manual]    {gse} | N={row['Samples']} | {row['Type']}")
            manual_targets.append({
                'Accession': gse,
                'Samples': row['Samples'],
                'Type': row['Type'],
                'URL': f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={gse}"
            })

    # Output action report for manual downloads
    if manual_targets:
        print("\n" + "-" * 65)
        print("ACTION REQUIRED: MANUAL DATA EXTRACTION (RNA-SEQ & COMPLEX)")
        print("-" * 65)
        print(f"Please download raw count matrices from 'Supplementary Files'.")
        print(f"Target directory: {SEQ_DIR}\n")
        
        for t in manual_targets:
            print(f"[{t['Accession']}] - N={t['Samples']} ({t['Type']})")
            print(f"Source: {t['URL']}\n")
    else:
        print("\n[*] All qualifying datasets were downloaded automatically.")
        
    print("[*] Data acquisition phase complete.")

# ==========================================
# EXECUTION GUARD
# ==========================================
if __name__ == "__main__":
    main()