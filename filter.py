"""
AIS Data Filter & Preparation Script (Versione 2025 Update)
-----------------------------------------------------------
Input  : CSV AIS grezzi (formato 2025 snake_case)
Output : file 'ais_subset.parquet' con nomi colonne STANDARD (MMSI, LAT, LON...)
"""

import pandas as pd
from glob import glob
import os

# === CONFIGURAZIONE PARAMETRI ===
DATA_DIR = "./input_raw/"        # cartella con i CSV estratti
LAT_MIN, LAT_MAX = 24, 30        # finestra geografica: Golfo del Messico
LON_MIN, LON_MAX = -93, -83
VESSEL_TYPE = 70                 # 70 = Cargo
N_SHIPS = 10                     # massimo MMSI da mantenere
OUTPUT_FILE = "./output/ais_subset_2025.parquet"

# === MAPPING COLONNE (Nuovo 2025 -> Standard Progetto) ===
# Questa mappa traduce i nomi nuovi in quelli che il tuo progetto si aspetta
COL_MAPPING = {
    'mmsi': 'MMSI',
    'base_date_time': 'BaseDateTime',
    'latitude': 'LAT',
    'longitude': 'LON',
    'sog': 'SOG',
    'cog': 'COG',
    'heading': 'Heading',
    'vessel_type': 'VesselType'
}

# Lista delle colonne nuove da caricare dal CSV
COLS_TO_LOAD = list(COL_MAPPING.keys())

# === 1. Caricamento file multipli ===
csv_files = sorted(glob(os.path.join(DATA_DIR, "*.csv")))
if not csv_files:
    raise FileNotFoundError("Nessun file CSV trovato nella cartella indicata.")

print(f"Trovati {len(csv_files)} file. Caricamento in corso...")
df_list = []

for f in csv_files:
    print("Leggo:", os.path.basename(f))
    
    # Lettura a chunk
    for chunk in pd.read_csv(f, chunksize=200000, low_memory=False):
        
        # 1. Filtra colonne esistenti nel chunk (per evitare errori se mancano)
        available_cols = [c for c in COLS_TO_LOAD if c in chunk.columns]
        chunk = chunk[available_cols]
        
        # 2. RINOMINA IMMEDIATA (Standardizzazione)
        # Trasformiamo 'latitude' -> 'LAT', 'mmsi' -> 'MMSI', ecc.
        chunk = chunk.rename(columns=COL_MAPPING)
        
        # 3. Ora usiamo i nomi STANDARD per il resto della logica
        # Verifica che ci siano le colonne essenziali dopo la rinomina
        required = ['LAT', 'LON', 'BaseDateTime', 'VesselType']
        if not all(col in chunk.columns for col in required):
            continue # Salta chunk se mancano dati critici
            
        chunk = chunk.dropna(subset=['LAT', 'LON', 'BaseDateTime'])
        
        # 4. Filtro Tipo Nave
        chunk = chunk[chunk['VesselType'] == VESSEL_TYPE]
        
        df_list.append(chunk)

if not df_list:
    raise ValueError("Nessun dato trovato dopo i filtri! Controlla VESSEL_TYPE o i file input.")

df = pd.concat(df_list, ignore_index=True)
print(f"Totale righe dopo unione e filtro tipo nave: {len(df):,}")

# === 2. Filtro geografico (Usa nomi STANDARD) ===
mask = (
    (df['LAT'] >= LAT_MIN) & (df['LAT'] <= LAT_MAX) &
    (df['LON'] >= LON_MIN) & (df['LON'] <= LON_MAX)
)
df = df[mask]
print(f"Righe dopo filtro area: {len(df):,}")

# === 3. Seleziona MMSI con più punti ===
if len(df) > 0:
    top_mmsi = df['MMSI'].value_counts().head(N_SHIPS).index
    df = df[df['MMSI'].isin(top_mmsi)]
    print(f"Tenuti i {N_SHIPS} MMSI più attivi.")
else:
    print("ATTENZIONE: Nessuna nave trovata nell'area geografica specificata.")

# === 4. Pulizia temporale e ordinamento ===
df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'], errors='coerce')
df = df.dropna(subset=['BaseDateTime'])
df = df.sort_values(['MMSI','BaseDateTime'])
print("Date ordinate e coerenti.")

# === 5. Salvataggio ===
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df = df.reset_index(drop=True)
df.to_parquet(OUTPUT_FILE, index=False)

print(f"Salvato file ridotto: {OUTPUT_FILE}")
print("Esempio righe (Formato Standardizzato):")
print(df.head())