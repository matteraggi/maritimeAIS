"""
Feature Engineering Avanzato (Physics-Enabled)
----------------------------------------------
Input  : './preprocessed/ais_preprocessed.parquet'
         (X,Y sono in METRI REALI. SOG,COG sono Normalizzati)
Output : './preprocessed/ais_final.parquet'
Scopo  : Calcolare derivate fisiche reali e statistiche mobili.
"""

import pandas as pd
import numpy as np
import os

# === CONFIG ===
INPUT_FILE = "./preprocessed/ais_preprocessed.parquet"
OUTPUT_FILE = "./preprocessed/ais_final.parquet"
WINDOW = 10  # Finestra di 10 minuti per le statistiche mobili

print(f"Carico dataset: {INPUT_FILE}")
df = pd.read_parquet(INPUT_FILE)
df = df.sort_values(["MMSI", "BaseDateTime"])

# === 1. Derivate Spaziali (ORA SONO REALI!) ===
# Poiché X e Y sono in metri, le differenze sono metri percorsi in 1 minuto.
print("Calcolo derivate spaziali (Metri/min)...")

# dX, dY: Spostamento in metri
df["dX"] = df.groupby("MMSI")["X"].diff()
df["dY"] = df.groupby("MMSI")["Y"].diff()

# speed_xy: Velocità scalare calcolata dalla posizione (Metri/minuto)
# Questa è la "Ground Truth" fisica, diversa dal SOG che è trasmesso.
df["speed_xy"] = np.sqrt(df["dX"]**2 + df["dY"]**2)

# === 2. Derivate Cinematiche (Segnale Radio) ===
# SOG e COG sono normalizzati, ma le loro variazioni indicano comunque accelerazioni/virate.
df["dSOG"] = df.groupby("MMSI")["SOG"].diff()
df["dCOG"] = df.groupby("MMSI")["COG"].diff()

# === 3. Statistiche Mobili (Rolling Features) ===
# Aggiungiamo anche speed_xy alle statistiche.
features_to_roll = ["SOG", "COG", "speed_xy", "dSOG", "dCOG"]

print(f"Calcolo rolling stats (finestra {WINDOW} min)...")
for col in features_to_roll:
    # Media mobile
    df[f"{col}_mean{WINDOW}"] = (
        df.groupby("MMSI")[col]
        .transform(lambda x: x.rolling(WINDOW, min_periods=1).mean())
    )
    # Deviazione standard (rilevamento manovre o anomalie)
    df[f"{col}_std{WINDOW}"] = (
        df.groupby("MMSI")[col]
        .transform(lambda x: x.rolling(WINDOW, min_periods=1).std())
    )

# === 4. Pulizia Finale ===
# I primi punti avranno NaN a causa del diff(). Li riempiamo a 0.
df = df.fillna(0)

# === 5. Salvataggio ===
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_parquet(OUTPUT_FILE, index=False)

# === 6. Log ===
print(f"Salvato: {OUTPUT_FILE}")
print(f"Righe finali: {len(df):,}")
print("Nuove feature fisiche calcolate:")
print(["dX", "dY", "speed_xy"])
print("Context features aggiunte:")
print([c for c in df.columns if "mean" in c or "std" in c])