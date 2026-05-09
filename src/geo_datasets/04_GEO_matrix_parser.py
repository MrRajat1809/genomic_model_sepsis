"""
04_GEO_matrix_parser.py

Master matrix and label parser for the elite sepsis datasets.
Extracts binary mortality labels using dataset-specific metadata keys.
Separately handles Microarray (GEO table extraction) and RNA-Seq (manual supplementary files)
matrix parsing, outputting compressed standardized matrices and a global label registry.
"""

import io
import gzip
import gc
import warnings
from pathlib import Path

import pandas as pd
import GEOparse

# Suppress GEOparse and Pandas warnings for cleaner terminal output
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
# Dynamically resolve paths from the script's location (src/geo_datasets/)
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
MA_DIR = DATA_DIR / "raw" / "microarray"
SEQ_DIR = DATA_DIR / "raw" / "rna-seq"
OUT_DIR = DATA_DIR / "processed" / "matrices"

# Ensure output directory exists
OUT_DIR.mkdir(parents=True, exist_ok=True)

# The 9 Elite Datasets
TARGET_DATASETS = {
    'GSE236713': {'dir': MA_DIR, 'target_key': 'died/survived', 'type': 'ma'},
    'GSE26440':  {'dir': MA_DIR, 'target_key': 'outcome', 'type': 'ma'},
    'GSE272769': {'dir': MA_DIR, 'target_key': 'mort30', 'type': 'ma'},
    'GSE54514':  {'dir': MA_DIR, 'target_key': 'disease status', 'type': 'ma'},
    'GSE65682':  {'dir': MA_DIR, 'target_key': 'mortality_event_28days', 'type': 'ma'},
    'GSE69063':  {'dir': MA_DIR, 'target_key': 'severity', 'type': 'ma'},
    'GSE95233':  {'dir': MA_DIR, 'target_key': 'survival', 'type': 'ma'},
    'GSE185263': {'dir': SEQ_DIR, 'target_key': 'in hospital mortality', 'type': 'rnaseq', 'file': 'GSE185263_raw_counts.csv.gz'},
    'GSE63042':  {'dir': SEQ_DIR, 'target_key': 'sirs outcomes', 'type': 'rnaseq', 'file': 'GSE63042_capsod_seq_rel_RPM_060314.xlsx.gz'}
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
    print("[*] INITIATING MASTER MATRIX AND LABEL PARSER (WITH RNA-SEQ FIX)...")

    master_labels = []

    for gse_id, config in TARGET_DATASETS.items():
        print(f"\n[+] Processing {gse_id}...")
        soft_path = config['dir'] / f"{gse_id}_family.soft.gz"
        
        if not soft_path.exists():
            print(f"    [!] Missing {soft_path.name}. Skipping.")
            continue

        try:
            # Load metadata only (cast Path to str for GEOparse)
            gse = GEOparse.get_GEO(filepath=str(soft_path), silent=True)
            valid_patients = 0
            target_key = config['target_key']
            
            # 1. Extract Labels
            for gsm_name, gsm in gse.gsms.items():
                chars = gsm.metadata.get('characteristics_ch1', [])
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
                    valid_patients += 1
                    
            print(f"    -> Extracted {valid_patients} valid mortality labels.")
            
            # 2. Extract Expression Matrices
            out_matrix_path = OUT_DIR / f"{gse_id}_matrix.csv.gz"
            
            # MICROARRAY PARSING
            if config['type'] == 'ma':
                print("    -> Extracting Microarray Matrix...")
                matrix_data = {}
                for gsm_name, gsm in gse.gsms.items():
                    if not gsm.table.empty:
                        if 'ID_REF' in gsm.table.columns and 'VALUE' in gsm.table.columns:
                            matrix_data[gsm_name] = gsm.table.set_index('ID_REF')['VALUE']
                            
                if matrix_data:
                    df_matrix = pd.DataFrame(matrix_data)
                    df_matrix.to_csv(out_matrix_path, compression='gzip')
                    print(f"    -> Matrix saved: {df_matrix.shape[0]} rows (probes) x {df_matrix.shape[1]} columns (patients)")
                    del df_matrix
            
            # RNA-SEQ PARSING
            elif config['type'] == 'rnaseq':
                print("    -> Extracting RNA-Seq Matrix...")
                file_path = config['dir'] / config['file']
                
                if file_path.exists():
                    if file_path.name.endswith('.csv.gz'):
                        df_matrix = pd.read_csv(file_path, index_col=0, compression='gzip')
                    elif file_path.name.endswith('.xlsx.gz'):
                        with gzip.open(file_path, 'rb') as f:
                            df_matrix = pd.read_excel(io.BytesIO(f.read()), index_col=0)
                    
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
    print(f"[*] Total Labeled Sepsis Patients Secured: {len(labels_df)}")
    print("="*50)

if __name__ == "__main__":
    main()