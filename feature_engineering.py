"""
Feature Engineering Avanzato (Optimized for AI Project)
-------------------------------------------------------
Input  : './preprocessed/ais_preprocessed.parquet' (X,Y sono Z-Scores)
Output : './preprocessed/ais_final.parquet'
Scopo  : Aggiungere contesto temporale (Rolling Stats) alle feature cinematiche.
Nota   : Non calcoliamo delta posizionali qui (dX, dY) perché X,Y sono normalizzati.
         I delta fisici in metri verranno calcolati nel Notebook di training.
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

# === 1. Derivate Cinematiche (Solo su grandezze fisiche indipendenti) ===
# Calcoliamo le variazioni di velocità e rotta.
# Anche se SOG/COG sono normalizzati, le loro differenze relative hanno senso per il pattern recognition.
df["dSOG"] = df.groupby("MMSI")["SOG"].diff()
df["dCOG"] = df.groupby("MMSI")["COG"].diff()

# Rate of Turn approssimato (dal COG)
df["TurnRate"] = df["dCOG"] 

# === 2. Statistiche Mobili (Rolling Features) ===
# Queste feature dicono alla rete: "Il comportamento è stabile o sta cambiando?"
features_to_roll = ["SOG", "COG", "dSOG", "dCOG"]

print(f"Calcolo rolling stats (finestra {WINDOW} min)...")
for col in features_to_roll:
    # Media mobile (Trend di breve periodo)
    df[f"{col}_mean{WINDOW}"] = (
        df.groupby("MMSI")[col]
        .transform(lambda x: x.rolling(WINDOW, min_periods=1).mean())
    )
    # Deviazione Standard mobile (Volatilità/Incertezza)
    # Utile per rilevare manovre improvvise o attacchi anomali
    df[f"{col}_std{WINDOW}"] = (
        df.groupby("MMSI")[col]
        .transform(lambda x: x.rolling(WINDOW, min_periods=1).std())
    )

# === 3. Pulizia Finale ===
# Riempie i NaN generati dai diff/rolling (i primi punti della sequenza) con 0
df = df.fillna(0)

# === 4. Salvataggio ===
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_parquet(OUTPUT_FILE, index=False)

# === 5. Log ===
print(f"Salvato: {OUTPUT_FILE}")
print(f"Righe finali: {len(df):,}")
print("Nuove colonne aggiunte (Context Features):")
print([c for c in df.columns if "mean" in c or "std" in c or "d" in c])