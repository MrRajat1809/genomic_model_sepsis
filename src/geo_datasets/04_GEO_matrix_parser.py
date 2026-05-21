"""
04_GEO_matrix_parser.py

Master matrix and label parser for the elite sepsis datasets.
MERGED VERSION:
1. Universal Metadata Parsing: Reads .soft.gz for BOTH Microarray and RNA-Seq to get labels.
2. Purity Filter: Surgically removes SIRS/Controls using patient-level attributes.
3. Rosetta Stone Translation: Maps author-specific RNA-Seq matrix IDs (e.g., 'sepcol001') 
   back to official NCBI 'GSM' IDs using metadata titles.
4. Matrix Slicing: Enforces strict dimensional alignment.
"""

import io
import gzip
import gc
import warnings
from pathlib import Path

import pandas as pd
import GEOparse

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
MA_DIR = DATA_DIR / "raw" / "microarray"
SEQ_DIR = DATA_DIR / "raw" / "rna-seq"
OUT_DIR = DATA_DIR / "processed" / "matrices"

OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_DATASETS = {
    'GSE236713': {'dir': MA_DIR, 'target_key': 'died/survived', 'type': 'ma', 'reject_keywords': ['sirs', 'control', 'healthy']},
    'GSE26440':  {'dir': MA_DIR, 'target_key': 'outcome', 'type': 'ma', 'reject_keywords': ['control', 'healthy']},
    'GSE272769': {'dir': MA_DIR, 'target_key': 'mort30', 'type': 'ma'},
    'GSE54514':  {'dir': MA_DIR, 'target_key': 'disease status', 'type': 'ma'},
    'GSE65682':  {'dir': MA_DIR, 'target_key': 'mortality_event_28days', 'type': 'ma'},
    'GSE95233':  {'dir': MA_DIR, 'target_key': 'survival', 'type': 'ma'},
    'GSE185263': {'dir': SEQ_DIR, 'target_key': 'in hospital mortality', 'type': 'rnaseq', 'file': 'GSE185263_raw_counts.csv.gz'}
}

# ==========================================
# HELPER FUNCTIONS
# ==========================================
def normalize_label(val: str):
    """Standardizes diverse cohort mortality labels into a binary format (1 = dead, 0 = survived)."""
    val = str(val).strip().lower()
    death_flags = ['died', 'nonsurvivor', 'non survivor', 'yes', '1', 'sepsis death', 'sepsis nonsurvivor']
    survival_flags = ['survived', 'survivor', 'no', '0', 'alive', 'sepsis survivor', 'uncomplicated sepsis', 'severe sepsis', 'septic shock']
    
    if val in death_flags: 
        return 1
    if val in survival_flags: 
        return 0
    return None

# ==========================================
# MAIN EXECUTION LOGIC
# ==========================================
def main():
    print("[*] INITIATING SURGICAL MATRIX AND LABEL PARSER...")

    master_labels = []

    for gse_id, config in TARGET_DATASETS.items():
        print(f"\n[+] Processing {gse_id}...")
        soft_path = config['dir'] / f"{gse_id}_family.soft.gz"
        
        if not soft_path.exists():
            print(f"    [!] Missing {soft_path.name}. Skipping.")
            continue

        try:
            valid_patients = []
            title_to_gsm = {}  # The Rosetta Stone dictionary
            target_key = config['target_key']
            
            # 1. UNIVERSAL LABEL EXTRACTION
            gse = GEOparse.get_GEO(filepath=str(soft_path), silent=True)
            
            for gsm_name, gsm in gse.gsms.items():
                # Memorize the author's original title just in case we need to translate later
                raw_title = str(gsm.metadata.get('title', [''])[0]).strip()
                title_to_gsm[raw_title] = gsm_name
                
                # Purity Check
                chars = gsm.metadata.get('characteristics_ch1', [])
                source = gsm.metadata.get('source_name_ch1', [])
                clinical_text = " ".join([str(x).lower() for x in chars + source])
                
                if 'reject_keywords' in config:
                    if any(keyword in clinical_text for keyword in config['reject_keywords']):
                        continue
                
                # Label Extraction
                patient_label = None
                for char in chars:
                    if ':' in char:
                        key, val = char.split(':', 1)
                        if key.strip().lower() == target_key.lower():
                            patient_label = normalize_label(val)
                            break
                            
                if patient_label is not None:
                    master_labels.append({
                        'Dataset': gse_id, 
                        'Patient_ID': gsm_name, 
                        'Mortality': patient_label
                    })
                    valid_patients.append(gsm_name)
            
            print(f"    -> Extracted {len(valid_patients)} pure mortality labels.")

            # 2. MATRIX CONSTRUCTION
            out_matrix_path = OUT_DIR / f"{gse_id}_matrix.csv.gz"

            # --- MICROARRAY ---
            if config['type'] == 'ma':
                matrix_data = {}
                for gsm_name in valid_patients:
                    gsm = gse.gsms[gsm_name]
                    if not gsm.table.empty and 'ID_REF' in gsm.table.columns and 'VALUE' in gsm.table.columns:
                        matrix_data[gsm_name] = gsm.table.set_index('ID_REF')['VALUE']
                
                if matrix_data:
                    df_matrix = pd.DataFrame(matrix_data)
                    df_matrix.to_csv(out_matrix_path, compression='gzip')
                    print(f"    -> Matrix saved: {df_matrix.shape[0]} rows (probes) x {df_matrix.shape[1]} columns (patients)")
                    del df_matrix
            
            # --- RNA-SEQ ---
            elif config['type'] == 'rnaseq':
                print("    -> Extracting RNA-Seq Matrix...")
                file_path = config['dir'] / config['file']
                
                if file_path.exists():
                    if file_path.name.endswith('.csv.gz'):
                        df_matrix = pd.read_csv(file_path, index_col=0, compression='gzip')
                    elif file_path.name.endswith('.xlsx.gz'):
                        with gzip.open(file_path, 'rb') as f:
                            df_matrix = pd.read_excel(io.BytesIO(f.read()), index_col=0)
                    
                    # Ensure column names are clean strings
                    df_matrix.columns = [str(c).strip() for c in df_matrix.columns]
                    
                    # THE ROSETTA STONE MAPPING
                    col_mapping = {}
                    for col in df_matrix.columns:
                        # Case 1: The column is already a proper GSM ID
                        for valid_gsm in valid_patients:
                            if valid_gsm in col:
                                col_mapping[col] = valid_gsm
                                break
                        
                        # Case 2: The column is the author's local title (e.g., 'sepcol001')
                        if col not in col_mapping:
                            for title, gsm_id in title_to_gsm.items():
                                if gsm_id in valid_patients and (col in title or title in col):
                                    col_mapping[col] = gsm_id
                                    break
                    
                    if not col_mapping:
                        print("    [!] FATAL: Could not map RNA-Seq columns to valid GSM IDs.")
                        print(f"    [!] Matrix columns: {list(df_matrix.columns[:5])}")
                    else:
                        # Rename columns to strict GSM IDs
                        df_matrix = df_matrix.rename(columns=col_mapping)
                        
                        # Slice down to ONLY the mapped, pure patients
                        final_cols = [c for c in df_matrix.columns if c in valid_patients]
                        df_matrix = df_matrix[final_cols]
                        
                        df_matrix.to_csv(out_matrix_path, compression='gzip')
                        print(f"    -> Matrix saved: {df_matrix.shape[0]} rows (genes) x {df_matrix.shape[1]} columns (patients)")
                    
                    del df_matrix
                else:
                    print(f"    [!] Missing manual file: {config['file']}")

            # Free RAM
            del gse
            gc.collect()

        except Exception as e:
            print(f"    [!] Error parsing {gse_id}: {e}")

    # 3. Export Global Label Registry
    labels_df = pd.DataFrame(master_labels)
    labels_out_path = OUT_DIR / "master_clinical_labels.csv"
    labels_df.to_csv(labels_out_path, index=False)

    print("\n" + "="*50)
    print("[*] MASTER PARSE COMPLETE.")
    print(f"[*] Total Purified Sepsis Patients Secured: {len(labels_df)}")
    print("="*50)

if __name__ == "__main__":
    main()