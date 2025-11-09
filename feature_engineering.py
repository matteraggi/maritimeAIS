"""
Feature Engineering Dinamiche AIS
---------------------------------
Input  : ./preprocessed/ais_preprocessed.parquet
Output : ./preprocessed/ais_final.parquet
Scopo  : aggiunge derivate cinematiche e statistiche mobili
"""

import pandas as pd
import numpy as np
import os

# === CONFIG ===
INPUT_FILE = "./preprocessed/ais_preprocessed.parquet"
OUTPUT_FILE = "./preprocessed/ais_final.parquet"
WINDOW = 10  # minuti per statistiche mobili (10 timestep)
EPS = 1e-9

print(f"Carico dataset: {INPUT_FILE}")
df = pd.read_parquet(INPUT_FILE)
df = df.sort_values(["MMSI", "BaseDateTime"])
df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])

# === 1. Derivate spaziali e cinematiche ===
df[["dX", "dY"]] = df.groupby("MMSI")[["X", "Y"]].diff()
df["dist_step"] = np.sqrt(df["dX"]**2 + df["dY"]**2)

# velocità calcolata da X,Y
df["speed_xy"] = df.groupby("MMSI")["dist_step"].transform(lambda x: x.fillna(0))
df["accel_xy"] = df.groupby("MMSI")["speed_xy"].diff()

# variazioni sugli altri parametri
df["dSOG"] = df.groupby("MMSI")["SOG"].diff()
df["dCOG"] = df.groupby("MMSI")["COG"].diff()
df["dHeading"] = df.groupby("MMSI")["Heading"].diff()

# tasso di virata (rotazione per passo temporale)
df["turn_rate"] = df["dHeading"]

# direzione di movimento
df["dir_xy"] = np.degrees(np.arctan2(df["dY"], df["dX"] + EPS))

# === 2. Statistiche mobili ===
for col in ["SOG", "COG", "Heading", "speed_xy"]:
    df[f"{col}_mean{WINDOW}"] = (
        df.groupby("MMSI")[col].transform(lambda x: x.rolling(WINDOW, min_periods=1).mean())
    )
    df[f"{col}_std{WINDOW}"] = (
        df.groupby("MMSI")[col].transform(lambda x: x.rolling(WINDOW, min_periods=1).std())
    )

# === 3. Pulizia finale ===
df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)

# === 4. Salvataggio ===
os.makedirs("preprocessed", exist_ok=True)
df.to_parquet(OUTPUT_FILE, index=False)

# === 5. Log finale ===
print(f"Salvato: {OUTPUT_FILE}")
print(f"Righe finali: {len(df):,}")
print("Colonne principali aggiunte:")
added = [c for c in df.columns if c not in ["MMSI", "BaseDateTime", "LAT", "LON", "X", "Y", "SOG", "COG", "Heading", "VesselType"]]
print(added)
