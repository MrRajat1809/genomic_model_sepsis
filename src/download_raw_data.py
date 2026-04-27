import os
import GEOparse

# Define the paths (since your root is mounted to /workspace)
BASE_DIR = "/workspace"
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")

# Create the data/raw directory if it doesn't exist yet
os.makedirs(RAW_DATA_DIR, exist_ok=True)

def fetch_geo_dataset(gse_id):
    print(f"[*] Fetching {gse_id} from NCBI GEO...")
    try:
        # get_GEO downloads the SOFT file which contains both clinical metadata and gene expression
        gse = GEOparse.get_GEO(geo=gse_id, destdir=RAW_DATA_DIR)
        print(f"[+] Successfully downloaded {gse_id}. It contains {len(gse.gsms)} samples.\n")
        return gse
    except Exception as e:
        print(f"[-] Error downloading {gse_id}: {e}\n")

if __name__ == "__main__":
    print(f"Target Directory: {RAW_DATA_DIR}\n")
    print("-" * 50)
    
    # 1. The Benchmark Cohort: Pediatric Sepsis (228 patients)
    fetch_geo_dataset("GSE66099")
    
    # 2. The Expansion Cohort: Adult Sepsis (802 patients)
    fetch_geo_dataset("GSE65682")
    
    print("-" * 50)
    print("Data collection phase complete. Check your data/raw/ folder!")