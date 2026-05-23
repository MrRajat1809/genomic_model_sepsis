"""
03_geo_matrix_parser.py

Unified matrix and label parser for target sepsis cohorts.
Performs the following operations:
1. Universal Metadata Parsing: Reads .soft.gz files across Microarray and RNA-Seq platforms.
2. Cohort Filtering: Excludes SIRS and healthy controls based on patient-level attributes.
3. ID Translation: Maps author-specific RNA-Seq matrix IDs to official NCBI GSM IDs.
4. Matrix Slicing: Enforces dimensional alignment between expression matrices and clinical labels.
"""

import gc
import gzip
import io
import warnings
from pathlib import Path

import GEOparse
import pandas as pd

# Suppress GEOparse output for clean logs
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
# Resolves to Multimodal_prediction_sepsis/
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
    """Standardizes heterogeneous clinical mortality labels into a binary format (1=Death, 0=Survival)."""
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
    """Main execution workflow."""
    print("[*] Initiating unified matrix and clinical label parsing...")

    master_labels = []

    for gse_id, config in TARGET_DATASETS.items():
        print(f"\n[+] Processing {gse_id}...")
        soft_path = config['dir'] / f"{gse_id}_family.soft.gz"
        
        if not soft_path.exists():
            print(f"    [!] Missing source file: {soft_path.name}. Skipping cohort.")
            continue

        try:
            valid_patients = []
            title_to_gsm = {}  # Cross-reference mapping dictionary
            target_key = config['target_key']
            
            # ---------------------------------------------------------
            # 1. UNIVERSAL LABEL EXTRACTION
            # ---------------------------------------------------------
            gse = GEOparse.get_GEO(filepath=str(soft_path), silent=True)
            
            for gsm_name, gsm in gse.gsms.items():
                # Store the author's original title for potential downstream mapping
                raw_title = str(gsm.metadata.get('title', [''])[0]).strip()
                title_to_gsm[raw_title] = gsm_name
                
                # Apply cohort exclusion criteria filtering
                chars = gsm.metadata.get('characteristics_ch1', [])
                source = gsm.metadata.get('source_name_ch1', [])
                clinical_text = " ".join([str(x).lower() for x in chars + source])
                
                if 'reject_keywords' in config:
                    if any(keyword in clinical_text for keyword in config['reject_keywords']):
                        continue
                
                # Isolate target clinical endpoint
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
            
            print(f"    -> Extracted {len(valid_patients)} validated mortality labels.")

            # ---------------------------------------------------------
            # 2. MATRIX CONSTRUCTION & FILTERING
            # ---------------------------------------------------------
            out_matrix_path = OUT_DIR / f"{gse_id}_matrix.csv.gz"

            # --- MICROARRAY PROCESSING ---
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
            
            # --- RNA-SEQ PROCESSING ---
            elif config['type'] == 'rnaseq':
                print("    -> Extracting RNA-Seq Matrix...")
                file_path = config['dir'] / config['file']
                
                if file_path.exists():
                    if file_path.name.endswith('.csv.gz'):
                        df_matrix = pd.read_csv(file_path, index_col=0, compression='gzip')
                    elif file_path.name.endswith('.xlsx.gz'):
                        with gzip.open(file_path, 'rb') as f:
                            df_matrix = pd.read_excel(io.BytesIO(f.read()), index_col=0)
                    
                    # Sanitize column types
                    df_matrix.columns = [str(c).strip() for c in df_matrix.columns]
                    
                    # RNA-Seq ID Cross-Reference Mapping
                    col_mapping = {}
                    for col in df_matrix.columns:
                        # Case 1: Column contains an exact GSM ID
                        for valid_gsm in valid_patients:
                            if valid_gsm in col:
                                col_mapping[col] = valid_gsm
                                break
                        
                        # Case 2: Column utilizes an author-specific local title
                        if col not in col_mapping:
                            for title, gsm_id in title_to_gsm.items():
                                if gsm_id in valid_patients and (col in title or title in col):
                                    col_mapping[col] = gsm_id
                                    break
                    
                    if not col_mapping:
                        print("    [ERROR] Matrix translation failed. Cannot map RNA-Seq columns to valid GSM IDs.")
                        print(f"    [INFO] Matrix columns: {list(df_matrix.columns[:5])}")
                    else:
                        # Standardize columns to strict GSM IDs
                        df_matrix = df_matrix.rename(columns=col_mapping)
                        
                        # Subset matrix to target patient population
                        final_cols = [c for c in df_matrix.columns if c in valid_patients]
                        df_matrix = df_matrix[final_cols]
                        
                        df_matrix.to_csv(out_matrix_path, compression='gzip')
                        print(f"    -> Matrix saved: {df_matrix.shape[0]} rows (genes) x {df_matrix.shape[1]} columns (patients)")
                    
                    del df_matrix
                else:
                    print(f"    [ERROR] Missing expected RNA-Seq raw file: {config['file']}")

            # Reallocate memory
            del gse
            gc.collect()

        except Exception as e:
            print(f"    [ERROR] Exception encountered while parsing {gse_id}: {e}")

    # ---------------------------------------------------------
    # 3. EXPORT GLOBAL CLINICAL REGISTRY
    # ---------------------------------------------------------
    labels_df = pd.DataFrame(master_labels)
    labels_out_path = OUT_DIR / "master_clinical_labels.csv"
    labels_df.to_csv(labels_out_path, index=False)

    print("\n" + "-" * 50)
    print("[*] Matrix parsing execution complete.")
    print(f"[*] Total validated sepsis patients processed: {len(labels_df)}")
    print("-" * 50)

# ==========================================
# EXECUTION GUARD
# ==========================================
if __name__ == "__main__":
    main()