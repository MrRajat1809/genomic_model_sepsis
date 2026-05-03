import os
import gzip
import gc
import pandas as pd
import GEOparse
import mygene
import numpy as np
import warnings
warnings.filterwarnings("ignore")

print("[*] INITIATING REINFORCED PROBE-TO-GENE TRANSLATOR...")

# Architecture
base_dir = "/workspace/data"
matrix_dir = os.path.join(base_dir, "processed/matrices")
soft_dir_ma = os.path.join(base_dir, "raw/microarray")
out_dir = os.path.join(base_dir, "processed/mapped_matrices")
os.makedirs(out_dir, exist_ok=True)

TARGETS = {
    'GSE236713': 'ma', 'GSE26440': 'ma', 'GSE272769': 'ma', 
    'GSE54514': 'ma', 'GSE65682': 'ma', 'GSE69063': 'ma', 'GSE95233': 'ma',
    'GSE185263': 'rnaseq', 'GSE63042': 'rnaseq'
}

def get_gpl_id(soft_path):
    try:
        with gzip.open(soft_path, 'rt', errors='ignore') as f:
            for line in f:
                if line.startswith('!Sample_platform_id'):
                    return line.split('=')[1].strip()
    except Exception as e:
        pass
    return None

def clean_symbol(val):
    if pd.isna(val) or val == '': return np.nan
    val = str(val).strip()
    
    if '//' in val:
        parts = [p.strip() for p in val.split('//')]
        for p in parts:
            if p != '---' and len(p) > 1 and not p.isdigit():
                return p.upper()
                
    if ';' in val:
        val = val.split(';')[0]
        
    if val == '---' or val.lower() == 'nan': return np.nan
    return val.upper()

def map_rnaseq_ensembl(index_list):
    print("      -> Ensembl IDs detected. Querying live MyGene API...")
    mg = mygene.MyGeneInfo()
    clean_ids = [str(x).split('.')[0] for x in index_list]
    results = mg.querymany(clean_ids, scopes='ensembl.gene', fields='symbol', species='human', verbose=False)
    
    mapping = {}
    for res in results:
        if 'symbol' in res and 'query' in res:
            mapping[res['query']] = res['symbol'].upper()
            
    return [mapping.get(x, np.nan) for x in clean_ids]

def map_entrez_to_symbol(index_list):
    print("      -> Querying live MyGene API to translate Entrez IDs...")
    mg = mygene.MyGeneInfo()
    clean_ids = [str(x).split('.')[0] for x in index_list]
    results = mg.querymany(clean_ids, scopes='entrezgene', fields='symbol', species='human', verbose=False)
    
    mapping = {}
    for res in results:
        if 'symbol' in res and 'query' in res:
            mapping[res['query']] = res['symbol'].upper()
            
    return [mapping.get(x, np.nan) for x in clean_ids]

for gse_id, tech_type in TARGETS.items():
    print(f"\n[+] Processing {gse_id} ({tech_type.upper()})...")
    matrix_path = os.path.join(matrix_dir, f"{gse_id}_matrix.csv.gz")
    
    if not os.path.exists(matrix_path):
        print(f"    [!] Matrix not found: {matrix_path}")
        continue

    try:
        print("    -> Loading matrix into RAM...")
        df = pd.read_csv(matrix_path, index_col=0, compression='gzip')
        initial_rows = df.shape[0]

        # -----------------------------------------------------
        # MICROARRAY MAPPING ROUTINE
        # -----------------------------------------------------
        if tech_type == 'ma':
            soft_path = os.path.join(soft_dir_ma, f"{gse_id}_family.soft.gz")
            gpl_id = get_gpl_id(soft_path)
            
            if not gpl_id:
                print("    [!] Could not determine GPL ID. Skipping.")
                continue
                
            print(f"    -> Platform detected: {gpl_id}. Downloading dictionary...")
            gpl = GEOparse.get_GEO(geo=gpl_id, destdir=soft_dir_ma, silent=True)
            gpl_table = gpl.table
            
            # Expanded search for symbol column
            symbol_col = None
            possible_cols = ['Gene Symbol', 'Symbol', 'SYMBOL', 'GENE_SYMBOL', 'gene_assignment', 'ILMN_Gene', 'GeneName', 'GENE_NAME', 'SystematicName', 'Gene_Symbol']
            
            for col in possible_cols:
                if col in gpl_table.columns:
                    symbol_col = col
                    break
                    
            if not symbol_col:
                # Fallback: Check for Entrez ID if Symbol is missing (The GSE69063 Fix)
                entrez_col = None
                for col in ['ENTREZ_GENE_ID', 'Entrez_Gene_ID', 'GeneID']:
                    if col in gpl_table.columns:
                        entrez_col = col
                        break
                
                if entrez_col:
                    print(f"    -> No Gene Symbol column. Falling back to Entrez mapping via '{entrez_col}'...")
                    entrez_dict = dict(zip(gpl_table['ID'], gpl_table[entrez_col]))
                    df.index = df.index.map(entrez_dict)
                    df = df[df.index.notna()]
                    df.index = map_entrez_to_symbol(df.index)
                else:
                    print(f"    [!] No known Gene Symbol OR Entrez column found in {gpl_id}.")
                    print(f"        Available Dictionary Columns: {list(gpl_table.columns)}")
                    continue
            else:
                print(f"    -> Mapping probes using column: '{symbol_col}'...")
                gpl_table['Clean_Symbol'] = gpl_table[symbol_col].apply(clean_symbol)
                probe_dict = dict(zip(gpl_table['ID'], gpl_table['Clean_Symbol']))
                df.index = df.index.map(probe_dict)

        # -----------------------------------------------------
        # RNA-SEQ MAPPING ROUTINE
        # -----------------------------------------------------
        elif tech_type == 'rnaseq':
            if any('ENSG' in str(idx) for idx in df.index[:10]):
                df.index = map_rnaseq_ensembl(df.index)
            else:
                print("      -> Gene Symbols detected natively.")
                df.index = [clean_symbol(x) for x in df.index]

        # -----------------------------------------------------
        # CLEANUP & AGGREGATION
        # -----------------------------------------------------
        print("    -> Coercing string artifacts to strict numeric format...")
        df = df.apply(pd.to_numeric, errors='coerce')
        
        print("    -> Aggregating duplicate genes and dropping unmapped probes...")
        df = df[df.index.notna()]
        df = df.groupby(df.index).mean()
        
        out_path = os.path.join(out_dir, f"{gse_id}_mapped.csv.gz")
        df.to_csv(out_path, compression='gzip')
        
        print(f"    -> SUCCESS: Shrunk from {initial_rows} probes to {df.shape[0]} unique Genes.")
        
        del df
        gc.collect()

    except Exception as e:
        print(f"    [!] Error processing {gse_id}: {e}")

print("\n" + "="*50)
print("[*] TRANSLATION COMPLETE. All matrices are now in HUGO Gene Symbol format.")
print(f"[*] Saved to: {out_dir}")
print("="*50)