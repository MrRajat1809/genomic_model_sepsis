"""
01_geo_dragnet.py

Queries the NCBI Gene Expression Omnibus (GEO) API to identify 
human whole-blood/PBMC transcriptomic datasets related to sepsis.
Extracts metadata for matching datasets, sorts them by sample size,
and compiles a master index CSV.
"""

import time
from pathlib import Path

import pandas as pd
import requests

# ==========================================
# CONFIGURATION
# ==========================================
EMAIL = "rkp6055@gmail.com"
SEARCH_QUERY = 'sepsis[Title] AND "Homo sapiens"[Organism] AND (blood[Description] OR PBMC[Description]) AND "gse"[Filter]'
GEO_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
GEO_SUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

BASE_DIR = Path(__file__).resolve().parents[2]
OUTPUT_DIR = BASE_DIR / "data" / "raw"
OUTPUT_PATH = OUTPUT_DIR / "geo_sepsis_master_menu.csv"

# ==========================================
# FUNCTIONS
# ==========================================
def query_geo_api(query: str, email: str, retmax: int = 500) -> list:
    """
    Executes a search query against the NCBI GEO database.
    
    Args:
        query (str): The search query string formatted for NCBI E-utilities.
        email (str): Developer email address (required by NCBI API).
        retmax (int): Maximum number of records to return.
        
    Returns:
        list: A list of unique GEO dataset IDs matching the query.
    """
    params = {
        "db": "gds",
        "term": query,
        "retmax": retmax,
        "retmode": "json",
        "email": email
    }
    
    response = requests.get(GEO_SEARCH_URL, params=params)
    response.raise_for_status()
    
    data = response.json()
    return data.get('esearchresult', {}).get('idlist', [])

def fetch_geo_metadata(id_list: list, email: str, batch_size: int = 50) -> pd.DataFrame:
    """
    Fetches detailed metadata for a list of GEO IDs in rate-limited batches.
    
    Args:
        id_list (list): List of GEO dataset IDs.
        email (str): Developer email address.
        batch_size (int): Number of IDs to process per API request.
        
    Returns:
        pd.DataFrame: Structured metadata containing Accession, Title, Samples, etc.
    """
    cohort_data = []
    
    for i in range(0, len(id_list), batch_size):
        batch_ids = id_list[i:i + batch_size]
        
        params = {
            "db": "gds",
            "id": ",".join(batch_ids),
            "retmode": "json",
            "email": email
        }
        
        response = requests.get(GEO_SUMMARY_URL, params=params)
        response.raise_for_status()
        sum_resp = response.json()
        
        if 'result' in sum_resp:
            for uid in batch_ids:
                if uid in sum_resp['result']:
                    doc = sum_resp['result'][uid]
                    
                    # Filter for independent Series (GSE)
                    if doc.get('entrytype') == 'GSE':
                        cohort_data.append({
                            'Accession': doc.get('accession', ''),
                            'Title': doc.get('title', ''),
                            'Samples': doc.get('n_samples', 0),
                            'Summary': doc.get('summary', ''),
                            'Type': doc.get('gdstype', '')
                        })
        
        print(f"    -> Processed {min(i + batch_size, len(id_list))} / {len(id_list)} datasets")
        time.sleep(1)  # Enforce NCBI rate limits
        
    return pd.DataFrame(cohort_data)

def main():
    """Main execution workflow."""
    print("[*] Querying NCBI API for datasets matching target criteria...")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Phase 1: Retrieve IDs
    id_list = query_geo_api(query=SEARCH_QUERY, email=EMAIL)
    print(f"[*] Search complete. Identified {len(id_list)} GEO datasets.")
    
    if not id_list:
        print("[!] No datasets found. Exiting.")
        return
        
    # Phase 2: Extract Metadata
    print("[*] Fetching metadata in rate-limited batches...")
    df_metadata = fetch_geo_metadata(id_list=id_list, email=EMAIL)
    print(f"[*] Metadata harvest complete. Retrieved {len(df_metadata)} valid Series (GSE) records.")
    
    # Phase 3: Format and Export
    if not df_metadata.empty:
        print("[*] Formatting and exporting data...")
        
        # Enforce numeric sorting
        df_metadata['Samples'] = pd.to_numeric(df_metadata['Samples'], errors='coerce')
        df_metadata = df_metadata.sort_values(by='Samples', ascending=False)
        
        # Export
        df_metadata.to_csv(OUTPUT_PATH, index=False)
        print(f"[+] Exported {len(df_metadata)} datasets to: {OUTPUT_PATH}")
        
        # Display verification
        print("\n[*] Top 10 largest datasets by sample size:")
        print(df_metadata[['Accession', 'Samples', 'Type']].head(10).to_string(index=False))
    else:
        print("[!] No metadata could be formatted. Exiting.")

# ==========================================
# EXECUTION GUARD
# ==========================================
if __name__ == "__main__":
    main()