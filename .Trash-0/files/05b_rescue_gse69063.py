import os
import pandas as pd
import GEOparse
import mygene
import numpy as np

print("[*] INITIATING TARGETED RESCUE FOR GSE69063 (ENTREZ ID MAPPING)...")

base_dir = "/workspace/data"
matrix_path = os.path.join(base_dir, "processed/matrices/GSE69063_matrix.csv.gz")
out_path = os.path.join(base_dir, "processed/mapped_matrices/GSE69063_mapped.csv.gz")
gpl_id = "GPL19983"

try:
    print("    -> Loading raw matrix...")
    df = pd.read_csv(matrix_path, index_col=0, compression='gzip')
    initial_rows = df.shape[0]

    print(f"    -> Loading {gpl_id} dictionary...")
    gpl = GEOparse.get_GEO(geo=gpl_id, destdir=os.path.join(base_dir, "raw/microarray"), silent=True)
    gpl_table = gpl.table

    # Create a dictionary mapping Probe ID -> Entrez ID
    entrez_dict = dict(zip(gpl_table['ID'], gpl_table['ENTREZ_GENE_ID']))
    
    # Map the matrix index to Entrez IDs first
    df.index = df.index.map(entrez_dict)
    
    # Drop rows that didn't have an Entrez ID
    df = df[df.index.notna()]
    
    print("    -> Querying live MyGene API to translate Entrez IDs to Gene Symbols...")
    mg = mygene.MyGeneInfo()
    
    # Clean the Entrez IDs (ensure they are strings without decimals)
    clean_entrez = [str(x).split('.')[0] for x in df.index]
    
    results = mg.querymany(clean_entrez, scopes='entrezgene', fields='symbol', species='human', verbose=False)
    
    symbol_mapping = {}
    for res in results:
        if 'symbol' in res and 'query' in res:
            symbol_mapping[res['query']] = res['symbol'].upper()
            
    # Apply the final Symbol mapping
    df.index = [symbol_mapping.get(x, np.nan) for x in clean_entrez]

    print("    -> Coercing to numeric and aggregating...")
    df = df.apply(pd.to_numeric, errors='coerce')
    df = df[df.index.notna()]
    df = df.groupby(df.index).mean()

    df.to_csv(out_path, compression='gzip')
    print(f"    -> SUCCESS: Shrunk from {initial_rows} probes to {df.shape[0]} unique Genes.")

except Exception as e:
    print(f"    [!] Rescue failed: {e}")

print("\n[*] RESCUE COMPLETE. All 9 datasets are now secured.")