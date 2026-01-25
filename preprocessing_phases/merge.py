import pandas as pd
from glob import glob
import os


# cartella dove hai i file filtrati
parquet_files = sorted(glob("./output/ais_subset*.parquet"))
print("File trovati:", parquet_files)

dfs = [pd.read_parquet(f) for f in parquet_files]
merged = pd.concat(dfs, ignore_index=True)

# rimuovi eventuali duplicati per sicurezza
merged = merged.drop_duplicates(subset=['MMSI','BaseDateTime'])

# ordina per nave e tempo
merged = merged.sort_values(['MMSI','BaseDateTime'])

# salva dataset unico
os.makedirs("./merged", exist_ok=True)

merged.to_parquet("./merged/ais_merged.parquet", index=False)
print("Salvato: ais_merged.parquet")
print(f"Totale righe: {len(merged):,}")
