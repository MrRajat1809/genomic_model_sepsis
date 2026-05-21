import os
import urllib.request
import gzip
import pandas as pd
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")

print("==================================================")
print("[*] PREPARING MULTIPLE EXTERNAL COHORTS")
print("[*] Targets: GSE57065 & GSE43277 (Platform-Agnostic)")
print("==================================================")

out_dir = "/workspace/data/processed/external_test"
os.makedirs(out_dir, exist_ok=True)

# The 20-Gene Panel
top_20 = ['TTC27', 'CCDC43', 'MPHOSPH10', 'MRPS2', 'EIF2B2', 'LIG4', 'PCSK9', 'ECSIT', 
          'PEX19', 'MRPL22', 'BTN3A2', 'APH1A', 'MYC', 'NOSIP', 'NNMT', 'HMOX1', 
          'ZER1', 'SLC5A9', 'SYMPK', 'NSMAF']

def get_platform_annot_url(gpl_id):
    """Calculates the dynamic NCBI folder structure based on the GPL ID length."""
    gpl_num = gpl_id.replace('GPL', '')
    if len(gpl_num) <= 3:
        folder = "GPLnnn"
    else:
        folder = f"GPL{gpl_num[:-3]}nnn"
    return f"https://ftp.ncbi.nlm.nih.gov/geo/platforms/{folder}/{gpl_id}/annot/{gpl_id}.annot.gz"

def download_and_parse_geo(geo_id):
    print(f"\n[*] Processing {geo_id}...")
    
    url = f"https://ftp.ncbi.nlm.nih.gov/geo/series/{geo_id[:-3]}nnn/{geo_id}/matrix/{geo_id}_series_matrix.txt.gz"
    matrix_file = os.path.join(out_dir, f"{geo_id}_series_matrix.txt.gz")
    
    if not os.path.exists(matrix_file):
        try:
            print(f"    -> Downloading Matrix...")
            urllib.request.urlretrieve(url, matrix_file)
        except Exception as e:
            print(f"[!] Failed to download {geo_id}: {e}")
            return None, None, None

    # --- EXTRACT METADATA AND PLATFORM ---
    metadata_lines = []
    platform_id = None
    with gzip.open(matrix_file, 'rt', errors='ignore') as f:
        for line in f:
            if line.startswith('!Sample_characteristics') or line.startswith('!Sample_title') or line.startswith('!Sample_source'):
                metadata_lines.append(line.strip().split('\t'))
            elif line.startswith('!Sample_platform_id') and not platform_id:
                platform_id = line.strip().split('\t')[1].replace('"', '')
                
    print(f"    -> Detected Platform: {platform_id}")

    meta_df = pd.DataFrame(metadata_lines).set_index(0).T
    if not meta_df.empty:
        meta_df.columns = [col.replace('!', '') for col in meta_df.columns]
    
    # NLP Extraction for Mortality (Expanded for new datasets)
    mortality_target = []
    # GSE57065 & GSE43277 store survival in characteristics or titles
    # We must also extract the Sample ID, which is typically stored as the index in our transposed meta_df
    # To reliably map Sample IDs, we need the !Sample_geo_accession row
    accession_lines = []
    with gzip.open(matrix_file, 'rt', errors='ignore') as f:
        for line in f:
            if line.startswith('!Sample_geo_accession'):
                accession_lines = line.strip().split('\t')[1:]
                break
                
    if not accession_lines:
        print("[!] Could not find Sample IDs.")
        return None, None, None

    # Re-parse metadata to align with accessions
    patient_data = {acc: "" for acc in accession_lines}
    with gzip.open(matrix_file, 'rt', errors='ignore') as f:
        for line in f:
            if line.startswith('!Sample_characteristics_ch1') or line.startswith('!Sample_title'):
                parts = line.strip().split('\t')[1:]
                for acc, val in zip(accession_lines, parts):
                    patient_data[acc] += f" {val.lower()} "

    for acc, chars in patient_data.items():
        # Look for death/survival indicators
        if any(w in chars for w in ['non-survivor', 'death', 'died', 'fatal', 'survival: 0', 'status: dead']):
            mortality_target.append((acc.replace('"', ''), 1))
        elif any(w in chars for w in ['survivor', 'alive', 'discharged', 'survival: 1', 'status: alive']):
            mortality_target.append((acc.replace('"', ''), 0))
            
    y_test = pd.DataFrame(mortality_target, columns=['Sample_ID', 'Mortality']).set_index('Sample_ID')
    print(f"    -> Extracted {len(y_test)} patients with explicit mortality labels.")
    
    if len(y_test) == 0:
        print("[!] Could not find standard mortality keywords in metadata. Skipping dataset.")
        return None, None, None

    print("    -> Parsing Expression Matrix...")
    expr_df = pd.read_csv(matrix_file, sep='\t', compression='gzip', comment='!', index_col=0)
    
    return expr_df, y_test, platform_id

def process_platform_annotation(gpl_id):
    annot_file = os.path.join(out_dir, f"{gpl_id}.annot.gz")
    if not os.path.exists(annot_file):
        print(f"    -> Downloading {gpl_id} Annotation...")
        annot_url = get_platform_annot_url(gpl_id)
        try:
            urllib.request.urlretrieve(annot_url, annot_file)
        except Exception as e:
            print(f"[!] Failed to download annotation: {e}")
            return None

    ids, genes = [], []
    id_idx, gene_idx = None, None
    with gzip.open(annot_file, 'rt', errors='ignore') as f:
        for line in f:
            if line.startswith('#') or line.startswith('^') or line.startswith('!'): continue
            parts = line.split('\t')
            if id_idx is None:
                header = [p.strip().upper() for p in parts]
                if 'ID' in header:
                    id_idx = header.index('ID')
                    # Match anything containing 'GENE SYMBOL' or 'GENE_SYMBOL'
                    gene_col = next((i for i, v in enumerate(header) if 'GENE SYMBOL' in v or 'GENE_SYMBOL' in v), None)
                    if gene_col is not None:
                        gene_idx = gene_col
                continue
            try:
                if id_idx is not None and gene_idx is not None:
                    probe_id = parts[id_idx].strip()
                    gene_sym = parts[gene_idx].strip()
                    if probe_id and gene_sym:
                        primary_gene = gene_sym.split('///')[0].strip()
                        ids.append(probe_id)
                        genes.append(primary_gene)
            except IndexError: pass
    
    if not ids:
        return None
    return pd.DataFrame({'ID': ids, 'Gene symbol': genes}).set_index('ID')

# 2. Process Datasets
for geo_id in ['GSE57065', 'GSE43277']:
    expr_df, y_test, platform_id = download_and_parse_geo(geo_id)
    if expr_df is None or len(y_test) == 0: continue
        
    annot_df = process_platform_annotation(platform_id)
    if annot_df is None:
        print("[!] Could not parse platform annotation.")
        continue
        
    print(f"    -> Mapping probes to Gene Symbols...")
    expr_mapped = expr_df.merge(annot_df, left_index=True, right_index=True).groupby('Gene symbol').max()
    
    X_test_raw = expr_mapped.T
    X_test_raw.index.name = 'Sample_ID'
    shared_samples = X_test_raw.index.intersection(y_test.index)
    X_test_raw = X_test_raw.loc[shared_samples]
    y_test = y_test.loc[shared_samples]
    
    for gene in top_20:
        if gene not in X_test_raw.columns: X_test_raw[gene] = 0.0
            
    X_test_filtered = X_test_raw[top_20]
    
    # Z-Score Normalization to eliminate Batch Effects
    scaler = StandardScaler()
    X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test_filtered), columns=X_test_filtered.columns, index=X_test_filtered.index)
    
    X_test_scaled.to_csv(os.path.join(out_dir, f"X_{geo_id.lower()}_external.csv"))
    y_test.to_csv(os.path.join(out_dir, f"y_{geo_id.lower()}_external.csv"))
    print(f"    -> [SUCCESS] Saved {geo_id} Tensors.")

print("\n==================================================")
print("[*] ALL EXTERNAL DATASETS PREPARED.")