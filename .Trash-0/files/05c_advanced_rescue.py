import os
import pandas as pd
import mygene
import gzip
import io
import warnings
warnings.filterwarnings("ignore")

print("[*] INITIATING ADVANCED DATASET RESCUE...")

base_dir = "/workspace/data"
matrix_dir = os.path.join(base_dir, "processed/matrices")
mapped_dir = os.path.join(base_dir, "processed/mapped_matrices")

# =====================================================================
# 1. FIX GSE272769: Translate RefSeq/Ensembl Transcripts
# =====================================================================
print("\n[+] Rescuing GSE272769 (RefSeq/Ensembl Transcripts)...")
try:
    df_272 = pd.read_csv(os.path.join(matrix_dir, "GSE272769_matrix.csv.gz"), index_col=0, compression='gzip')
    mg = mygene.MyGeneInfo()
    clean_ids = [str(x).split('.')[0] for x in df_272.index]
    
    print("    -> Querying live MyGene API for Transcript scopes...")
    # Scopes updated to look specifically for Transcripts and Accession numbers
    res_272 = mg.querymany(clean_ids, scopes='refseq,ensembltranscript,accession.transcript', fields='symbol', species='human', verbose=False)
    
    mapping_272 = {r['query']: r['symbol'].upper() for r in res_272 if 'symbol' in r and 'query' in r}
    df_272.index = [mapping_272.get(x, None) for x in clean_ids]
    
    df_272 = df_272[df_272.index.notna()]
    df_272 = df_272.apply(pd.to_numeric, errors='coerce').groupby(df_272.index).mean()
    df_272.to_csv(os.path.join(mapped_dir, "GSE272769_mapped.csv.gz"), compression='gzip')
    print(f"    -> SUCCESS: Secured {df_272.shape[0]} valid Gene Symbols.")
except Exception as e:
    print(f"    [!] Failed to rescue GSE272769: {e}")


# =====================================================================
# 2. FIX GSE63042: Extract Hidden Gene Column from Excel
# =====================================================================
print("\n[+] Rescuing GSE63042 (Hunting for hidden Gene Symbol column)...")
try:
    excel_path = os.path.join(base_dir, "raw/rna-seq/GSE63042_capsod_seq_rel_RPM_060314.xlsx.gz")
    with gzip.open(excel_path, 'rb') as f:
        # Read without setting index first so we can search all columns
        df_630 = pd.read_excel(io.BytesIO(f.read()))
    
    # Cufflinks usually outputs 'gene_short_name', 'symbol', or 'gene'
    symbol_col = next((c for c in df_630.columns if any(x in str(c).lower() for x in ['gene_short_name', 'symbol', 'gene', 'name'])), None)
    
    if symbol_col:
        print(f"    -> Found hidden Gene Symbols in column: '{symbol_col}'")
        df_630.index = df_630[symbol_col].astype(str).str.upper()
        
        # Drop metadata columns and keep only pure numbers (patient expression)
        df_630 = df_630.select_dtypes(include=['number'])
        
        # Clean and aggregate
        df_630 = df_630[df_630.index != 'NAN']
        df_630 = df_630.groupby(df_630.index).mean()
        
        df_630.to_csv(os.path.join(mapped_dir, "GSE63042_mapped.csv.gz"), compression='gzip')
        print(f"    -> SUCCESS: Secured {df_630.shape[0]} valid Gene Symbols.")
    else:
        print(f"    [!] Could not auto-detect symbol column. Available: {list(df_630.columns)}")
except Exception as e:
    print(f"    [!] Failed to rescue GSE63042: {e}")


# =====================================================================
# 3. DIAGNOSE GSE185263: The Mismatched Patient IDs
# =====================================================================
print("\n[+] Diagnosing GSE185263 Mismatch...")
try:
    df_185 = pd.read_csv(os.path.join(mapped_dir, "GSE185263_mapped.csv.gz"), index_col=0, compression='gzip', nrows=2)
    matrix_cols = list(df_185.columns)[:5]
    
    labels_df = pd.read_csv(os.path.join(base_dir, "processed/matrices/master_clinical_labels.csv"))
    gse185_labels = labels_df[labels_df['Dataset'] == 'GSE185263']['Patient_ID'].tolist()[:5]
    
    print(f"    -> Matrix Headers (What the matrix uses): {matrix_cols}")
    print(f"    -> Clinical Labels (What GEO metadata uses): {gse185_labels}")
except Exception as e:
    print(f"    [!] Failed to diagnose GSE185263: {e}")

print("\n[*] RESCUE OPERATIONS COMPLETE.")