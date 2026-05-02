import os
import gzip
import gc
import pandas as pd
import GEOparse
import warnings
warnings.filterwarnings("ignore")

print("[*] INITIATING MASTER MATRIX AND LABEL PARSER (WITH RNA-SEQ FIX)...")

# Architecture
base_dir = "/workspace/data"
ma_dir = os.path.join(base_dir, "raw/microarray")
seq_dir = os.path.join(base_dir, "raw/rna-seq")
out_dir = os.path.join(base_dir, "processed/matrices")
os.makedirs(out_dir, exist_ok=True)

# The 9 Elite Datasets
TARGET_DATASETS = {
    'GSE236713': {'dir': ma_dir, 'target_key': 'died/survived', 'type': 'ma'},
    'GSE26440':  {'dir': ma_dir, 'target_key': 'outcome', 'type': 'ma'},
    'GSE272769': {'dir': ma_dir, 'target_key': 'mort30', 'type': 'ma'},
    'GSE54514':  {'dir': ma_dir, 'target_key': 'disease status', 'type': 'ma'},
    'GSE65682':  {'dir': ma_dir, 'target_key': 'mortality_event_28days', 'type': 'ma'},
    'GSE69063':  {'dir': ma_dir, 'target_key': 'severity', 'type': 'ma'},
    'GSE95233':  {'dir': ma_dir, 'target_key': 'survival', 'type': 'ma'},
    'GSE185263': {'dir': seq_dir, 'target_key': 'in hospital mortality', 'type': 'rnaseq', 'file': 'GSE185263_raw_counts.csv.gz'},
    'GSE63042':  {'dir': seq_dir, 'target_key': 'sirs outcomes', 'type': 'rnaseq', 'file': 'GSE63042_capsod_seq_rel_RPM_060314.xlsx.gz'}
}

def normalize_label(val):
    val = str(val).strip().lower()
    death_flags = ['died', 'nonsurvivor', 'non survivor', 'yes', '1', 'sepsis death', 'sepsis nonsurvivor']
    survival_flags = ['survived', 'survivor', 'no', '0', 'alive', 'sepsis survivor', 'uncomplicated sepsis', 'severe sepsis', 'septic shock']
    if val in death_flags: return 1
    if val in survival_flags: return 0
    return None

master_labels = []

for gse_id, config in TARGET_DATASETS.items():
    print(f"\n[+] Processing {gse_id}...")
    soft_path = os.path.join(config['dir'], f"{gse_id}_family.soft.gz")
    
    if not os.path.exists(soft_path):
        print(f"    [!] Missing {soft_path}. Skipping.")
        continue

    try:
        # Load metadata only
        gse = GEOparse.get_GEO(filepath=soft_path, silent=True)
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
                master_labels.append({'Dataset': gse_id, 'Patient_ID': gsm_name, 'Mortality': patient_label})
                valid_patients += 1
                
        print(f"    -> Extracted {valid_patients} valid mortality labels.")
        
        # 2. Extract Expression Matrices
        out_matrix_path = os.path.join(out_dir, f"{gse_id}_matrix.csv.gz")
        
        # MICROARRAY PARSING (Your original working code)
        if config['type'] == 'ma':
            print(f"    -> Extracting Microarray Matrix...")
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
        
        # RNA-SEQ PARSING (The Fix)
        elif config['type'] == 'rnaseq':
            print(f"    -> Extracting RNA-Seq Matrix...")
            file_path = os.path.join(config['dir'], config['file'])
            if os.path.exists(file_path):
                if file_path.endswith('.csv.gz'):
                    df_matrix = pd.read_csv(file_path, index_col=0, compression='gzip')
                elif file_path.endswith('.xlsx.gz'):
                    import io
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

labels_df = pd.DataFrame(master_labels)
labels_out_path = os.path.join(out_dir, "master_clinical_labels.csv")
labels_df.to_csv(labels_out_path, index=False)

print("\n" + "="*50)
print(f"[*] MASTER PARSE COMPLETE.")
print(f"[*] Total Labeled Sepsis Patients Secured: {len(labels_df)}")
print("="*50)