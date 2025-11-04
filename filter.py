"""
AIS Data Filter & Preparation Script (Cargo + Golfo del Messico)
----------------------------------------------------------------
Input  : più CSV AIS grezzi (es. AIS_2024_01_01.csv, AIS_2024_01_02.csv, ...)
Output : file 'ais_subset.parquet' con Cargo nel Golfo del Messico
Scopo  : ridurre i dati da centinaia di MB a un dataset gestibile per ML
"""

import pandas as pd
from glob import glob
import os

# === CONFIGURAZIONE PARAMETRI ===
DATA_DIR = "./input_raw/"        # cartella con i CSV estratti
LAT_MIN, LAT_MAX = 24, 30        # finestra geografica: Golfo del Messico
LON_MIN, LON_MAX = -93, -83
VESSEL_TYPE = 52                 # tipo nave: Cargo
N_SHIPS = 10                     # massimo MMSI da mantenere
OUTPUT_FILE = "./output/ais_subset2.parquet"

# === 1. Caricamento file multipli ===
csv_files = sorted(glob(os.path.join(DATA_DIR, "*.csv")))
if not csv_files:
    raise FileNotFoundError("Nessun file CSV trovato nella cartella indicata.")

print(f"Trovati {len(csv_files)} file. Caricamento in corso...")
df_list = []
for f in csv_files:
    print("Leggo:", os.path.basename(f))
    # Lettura a chunk per evitare problemi di memoria
    for chunk in pd.read_csv(f, chunksize=200000, low_memory=False):
        # tieni solo le colonne necessarie se esistono
        cols = [c for c in ['MMSI','BaseDateTime','LAT','LON','SOG','COG','Heading','VesselType'] if c in chunk.columns]
        chunk = chunk[cols].dropna(subset=['LAT','LON','BaseDateTime'])
        # filtro per tipo nave
        if 'VesselType' in chunk.columns:
            chunk = chunk[chunk['VesselType'] == VESSEL_TYPE]
        df_list.append(chunk)

df = pd.concat(df_list, ignore_index=True)
print(f"Totale righe dopo unione e filtro tipo nave: {len(df):,}")

# === 2. Filtro geografico ===
mask = (
    (df['LAT'] >= LAT_MIN) & (df['LAT'] <= LAT_MAX) &
    (df['LON'] >= LON_MIN) & (df['LON'] <= LON_MAX)
)
df = df[mask]
print(f"Righe dopo filtro area [{LAT_MIN},{LAT_MAX}]x[{LON_MIN},{LON_MAX}]: {len(df):,}")

# === 3. Seleziona MMSI con più punti ===
top_mmsi = df['MMSI'].value_counts().head(N_SHIPS).index
df = df[df['MMSI'].isin(top_mmsi)]
print(f"Tenuti i {N_SHIPS} MMSI più attivi.")

# === 4. Pulizia temporale e ordinamento ===
df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], errors='coerce')
df = df.dropna(subset=['BaseDateTime'])
df = df.sort_values(['MMSI','BaseDateTime'])
print("Date ordinate e coerenti.")

# === 5. Salvataggio ===
df = df.reset_index(drop=True)
df.to_parquet(OUTPUT_FILE, index=False)
print(f"Salvato file ridotto: {OUTPUT_FILE}")
print("Esempio righe:")
print(df.head())

"""
OUTPUT
------
- 'ais_subset.parquet' (pochi MB)
- colonne: MMSI, BaseDateTime, LAT, LON, SOG, COG, Heading, VesselType
- dati: Cargo (52) nel Golfo del Messico
- pronto per preprocessing AI
"""
