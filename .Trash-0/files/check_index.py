import pandas as pd
base = "/workspace/data/processed/mapped_matrices/"
# Load a "good" one and the "broken" one
df_good = pd.read_csv(base + "GSE54514_mapped.csv.gz", index_col=0, nrows=5)
df_bad = pd.read_csv(base + "GSE63042_mapped.csv.gz", index_col=0, nrows=5)

print("--- GOOD GENES (GSE54514) ---")
print(df_good.index.tolist())
print("\n--- BROKEN GENES (GSE63042) ---")
print(df_bad.index.tolist())