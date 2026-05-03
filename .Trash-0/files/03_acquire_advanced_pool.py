import os
import urllib.request
from tqdm import tqdm

print("[*] INITIATING ADVANCED DATABASE ACQUISITION (GEO + ARRAYEXPRESS)...")

data_dir = "/workspace/data/raw/advanced_pool"
os.makedirs(data_dir, exist_ok=True)

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
        print(f"    -> Downloading {desc}...")
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
            urllib.request.urlretrieve(url, file_path, reporthook=t.update_to)
    except Exception as e:
        print(f"    [!] Failed to download {desc}: {e}")

# 1. New GEO Targets (RNA-Seq and Trauma sets)
geo_targets = ['GSE185263', 'GSE12624']

for gse in geo_targets:
    prefix = gse[:-3] + "nnn"
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{prefix}/{gse}/soft/{gse}_family.soft.gz"
    dest = os.path.join(data_dir, f"{gse}_family.soft.gz")
    download_file(url, dest, gse)

# 2. ArrayExpress Target Example (E-MEXP-3567)
# For ArrayExpress, we pull the SDRF (Sample Data Relationship Format) which holds the clinical labels
ae_targets = ['E-MEXP-3567']

for ae_id in ae_targets:
    # ArrayExpress metadata URL structure
    sdrf_url = f"https://www.ebi.ac.uk/arrayexpress/files/{ae_id}/{ae_id}.sdrf.txt"
    dest = os.path.join(data_dir, f"{ae_id}_metadata.sdrf.txt")
    download_file(sdrf_url, dest, f"{ae_id} Metadata")

print("\n[*] ADVANCED DOWNLOAD COMPLETE.")