import os
import pandas as pd
import GEOparse
import gc
import warnings
warnings.filterwarnings("ignore")

print("[*] INITIATING SECURE RESCUE PROTOCOL...")

base_dir = "/workspace/data"
matrix_dir = os.path.join(base_dir, "processed/matrices")
mapped_dir = os.path.join(base_dir, "processed/mapped_matrices")
soft_dir_ma = os.path.join(base_dir, "raw/microarray")
soft_dir_seq = os.path.join(base_dir, "raw/rna-seq")

# =====================================================================
# 1. PATCH GSE185263: The Metadata Crosswalk
# =====================================================================
print("\n[+] Patching GSE185263 (Metadata Crosswalk)...")
try:
    # 1. Parse the SOFT file to get the true mapping
    soft_path = os.path.join(soft_dir_seq, "GSE185263_family.soft.gz")
    gse = GEOparse.get_GEO(filepath=soft_path, silent=True)
    
    # Map GSM ID -> Title (The title usually contains the 'sepcol' name)
    gsm_to_title = {}
    for gsm_name, gsm in gse.gsms.items():
        title = gsm.metadata.get('title', [''])[0].strip().lower()
        gsm_to_title[title] = gsm_name
        
    del gse
    gc.collect()

    # 2. Load the matrix
    df_185 = pd.read_csv(os.path.join(mapped_dir, "GSE185263_mapped.csv.gz"), index_col=0, compression='gzip')
    
    # 3. Rename columns by matching the 'sepcol' name to the GEO title
    new_columns = []
    for col in df_185.columns:
        clean_col = str(col).strip().lower()
        # Find if this column name is hidden inside any of the GEO titles
        matching_gsm = next((gsm for title, gsm in gsm_to_title.items() if clean_col in title), col)
        new_columns.append(matching_gsm)
        
    df_185.columns = new_columns
    
    # Save the properly named matrix
    df_185.to_csv(os.path.join(mapped_dir, "GSE185263_mapped.csv.gz"), compression='gzip')
    print(f"    -> SUCCESS: Matrix columns successfully translated to GSM IDs.")
    
    del df_185
    gc.collect()
except Exception as e:
    print(f"    [!] Failed to patch GSE185263: {e}")

# =====================================================================
# 2. PATCH GSE272769: The Affymetrix String Parser
# =====================================================================
print("\n[+] Patching GSE272769 (Affymetrix string bug)...")
try:
    df_272 = pd.read_csv(os.path.join(matrix_dir, "GSE272769_matrix.csv.gz"), index_col=0, compression='gzip')
    
    gpl = GEOparse.get_GEO(geo="GPL17692", destdir=soft_dir_ma, silent=True)
    gpl_table = gpl.table
    
    def get_affy_symbol(val):
        if pd.isna(val): return None
        parts = str(val).split('//')
        if len(parts) > 1:
            symbol = parts[1].strip()
            if symbol != '---': return symbol.upper()
        return None
        
    gpl_table['True_Symbol'] = gpl_table['gene_assignment'].apply(get_affy_symbol)
    probe_dict = dict(zip(gpl_table['ID'], gpl_table['True_Symbol']))
    
    df_272.index = df_272.index.map(probe_dict)
    df_272 = df_272[df_272.index.notna()]
    df_272 = df_272.apply(pd.to_numeric, errors='coerce').groupby(df_272.index).mean()
    
    df_272.to_csv(os.path.join(mapped_dir, "GSE272769_mapped.csv.gz"), compression='gzip')
    print(f"    -> SUCCESS: Affymetrix strings parsed. Secured {df_272.shape[0]} valid Gene Symbols.")
    
    del df_272
    del gpl
    gc.collect()
except Exception as e:
    print(f"    [!] Failed to patch GSE272769: {e}")

print("\n[*] RESCUE COMPLETE. FLUSHING RAM...")
gc.collect()