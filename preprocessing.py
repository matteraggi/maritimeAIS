"""
AIS Preprocessing Script (Optimized)
------------------------------------
Input  : 'ais_subset_2025.parquet'
Output : 'ais_preprocessed.parquet' + JSON Stats
Miglioramenti: Interpolazione limitata, gestione tipi numerici, pulizia Heading
"""

import pandas as pd
import numpy as np
from pyproj import Transformer
import os
import json

# === CONFIGURAZIONE ===
INPUT_FILE = "./output/ais_subset_2025.parquet"
OUTPUT_DIR = "./preprocessed"
OUTPUT_FILE = f"{OUTPUT_DIR}/ais_preprocessed.parquet"
MEAN_FILE = f"{OUTPUT_DIR}/feature_means.json"
STD_FILE  = f"{OUTPUT_DIR}/feature_stds.json"

DELTA_T = "1min"          # Intervallo temporale fisso
EPSG = 32616              # UTM zona 16N (Golfo del Messico)
MAX_INTERPOLATION_MIN = 30 # NON inventare dati se il buco > 30 min

# === 1. Caricamento ===
if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(f"File non trovato: {INPUT_FILE}")

df = pd.read_parquet(INPUT_FILE)
print(f"Righe iniziali: {len(df):,}")

# === 2. Conversione coordinate (Lat/Lon -> Metri Locali) ===
print("Proiezione coordinate in Metri (UTM)...")
transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True)
df["X"], df["Y"] = transformer.transform(df["LON"].values, df["LAT"].values)

# === 3. Interpolazione temporale (con LIMITE) ===
print(f"Resampling a {DELTA_T} con limite interpolazione {MAX_INTERPOLATION_MIN} min...")

def resample_ship(group):
    # Ordina e indicizza per tempo
    g = group.sort_values("BaseDateTime").set_index("BaseDateTime")
    
    # 1. Resample (crea la griglia dei minuti vuoti)
    # numeric_only=True evita errori con colonne stringa
    g_res = g.resample(DELTA_T).mean(numeric_only=True)
    
    # 2. Interpolazione LIMITATA
    # Se manca il segnale per più di 30 min, lascia NaN (spezza la sequenza)
    g_res = g_res.interpolate(method='linear', limit=MAX_INTERPOLATION_MIN)
    
    # Ripristina MMSI (perso durante il mean)
    g_res["MMSI"] = group["MMSI"].iloc[0]
    
    # Rimuovi le righe che sono rimaste NaN (buchi troppo grandi)
    return g_res.dropna().reset_index()

df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])
# group_keys=False evita indici doppi
df_interp = df.groupby("MMSI", group_keys=False).apply(resample_ship)

print(f"Righe dopo interpolazione e pulizia buchi: {len(df_interp):,}")

# === 4. Pulizia valori fisici ===
# Teniamo solo valori fisicamente sensati
df_interp = df_interp[
    (df_interp["SOG"].between(0, 50)) &  # Speed (nodi)
    (df_interp["COG"].between(0, 360))   # Course (gradi)
].copy()

# === 5. Normalizzazione ===
# Usiamo solo le feature che servono alla rete. Rimuoviamo Heading (spesso rumoroso).
numeric_cols = ["X", "Y", "SOG", "COG"]

# Calcola statistiche
stats = df_interp[numeric_cols].describe()
means = stats.loc["mean"].to_dict()
stds = stats.loc["std"].to_dict()

# Applica normalizzazione (Z-Score)
# Nota: X e Y vengono normalizzati qui. Nel Notebook li denormalizzeremo 
# usando i JSON per calcolare i delta in metri.
df_norm = df_interp.copy()
for col in numeric_cols:
    # Evita divisione per zero se std è 0 (caso raro ma possibile)
    if stds[col] > 0:
        df_norm[col] = (df_norm[col] - means[col]) / stds[col]
    else:
        df_norm[col] = 0.0

# === 6. Salvataggio ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Salva JSON
with open(MEAN_FILE, "w") as f: json.dump(means, f)
with open(STD_FILE, "w") as f: json.dump(stds, f)
print("✅ Salvate statistiche JSON.")

# Salva Parquet
df_norm = df_norm.sort_values(["MMSI", "BaseDateTime"]).reset_index(drop=True)
df_norm.to_parquet(OUTPUT_FILE, index=False)

print(f"✅ Dataset preprocessato salvato: {OUTPUT_FILE}")
print(f"Righe finali: {len(df_norm):,}")
print("Esempio dati normalizzati:")
print(df_norm[["MMSI", "X", "Y", "SOG", "COG"]].head())