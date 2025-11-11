"""
AIS Preprocessing Script
------------------------
Input  : 'ais_merged.parquet' (Cargo, Golfo del Messico)
Output : 'ais_preprocessed.parquet'
Scopo  : uniformare intervalli temporali, convertire coordinate,
         normalizzare e preparare sequenze per ML
"""

import pandas as pd
import numpy as np
from pyproj import Proj, Transformer
import os

# === CONFIGURAZIONE ===
INPUT_FILE = "./merged/ais_merged.parquet"
OUTPUT_FILE = "./preprocessed/ais_preprocessed.parquet"
DELTA_T = "1min"          # intervallo temporale uniforme
SEQUENCE_LEN = 60         # punti per sequenza (60 minuti)
REF_LAT, REF_LON = 27, -88  # centro area per proiezione locale
EPSG = 32616              # UTM zona 16N (Golfo del Messico)

# === 1. Caricamento ===
df = pd.read_parquet(INPUT_FILE)
print(f"Righe iniziali: {len(df):,}")

# === 2. Conversione coordinate (lat/lon -> metri locali) ===
transformer = Transformer.from_crs("EPSG:4326", f"EPSG:{EPSG}", always_xy=True)
df["X"], df["Y"] = transformer.transform(df["LON"].values, df["LAT"].values)

# === 3. Interpolazione temporale per ogni nave ===
def resample_ship(group):
    g = group.sort_values("BaseDateTime").set_index("BaseDateTime")
    g = g.resample(DELTA_T).mean().interpolate()
    g["MMSI"] = group["MMSI"].iloc[0]
    return g.reset_index()

df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])
df_interp = df.groupby("MMSI", group_keys=False).apply(resample_ship)
print(f"Righe dopo interpolazione: {len(df_interp):,}")

# === 4. Pulizia valori fisici ===
df_interp = df_interp[
    (df_interp["SOG"].between(0, 40)) &  # speed over ground in nodi
    (df_interp["COG"].between(0, 360))
]

# === 5. Normalizzazione ===
# === 5. Normalizzazione e salvataggio statistiche ===
numeric_cols = ["X", "Y", "SOG", "COG", "Heading"]

# Calcola statistiche
stats = df_interp[numeric_cols].describe()
means = stats.loc["mean"].to_dict()
stds = stats.loc["std"].to_dict()

# Applica normalizzazione
df_norm = df_interp.copy()
for col in numeric_cols:
    df_norm[col] = (df_norm[col] - means[col]) / stds[col]

# Salva statistiche per denormalizzazione futura
os.makedirs("preprocessed", exist_ok=True)
pd.Series(means).to_json("preprocessed/feature_means.json")
pd.Series(stds).to_json("preprocessed/feature_stds.json")
print("✅ Salvate statistiche in preprocessed/feature_means.json e feature_stds.json")

# === 6. Creazione finestre sequenziali ===
# (qui solo salvataggio lineare; il batching lo farai su Colab)
df_norm = df_norm.sort_values(["MMSI", "BaseDateTime"]).reset_index(drop=True)

os.makedirs("preprocessed", exist_ok=True)
df_norm.to_parquet(OUTPUT_FILE, index=False)
print(f"Salvato: {OUTPUT_FILE}")
print(f"Righe finali: {len(df_norm):,}")
print("Esempio righe:")
print(df_norm.head())

"""
OUTPUT
------
File: preprocessed/ais_preprocessed.parquet
Colonne: MMSI, BaseDateTime, X, Y, SOG, COG, Heading (normalizzate)
Usalo su Colab per addestrare LSTM/LNN con sequenze di 60 punti.
"""
 