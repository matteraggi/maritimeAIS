"""
Feature Engineering Avanzato (Physics-Enabled & Time-Aware)
-----------------------------------------------------------
Input  : './preprocessed/ais_preprocessed.parquet'
Output : './preprocessed/ais_feature_engineered.parquet'
Scopo  : 
  1. Identificare i "Viaggi" (Trip_ID) basandosi sui buchi temporali.
  2. Calcolare derivate fisiche SOLO all'interno dello stesso viaggio.
  3. Evitare velocità ipersoniche dovute a salti temporali.
"""

import pandas as pd
import numpy as np
import os

# === CONFIG ===
INPUT_FILE = "./preprocessed/ais_preprocessed.parquet"
OUTPUT_FILE = "./preprocessed/ais_feature_engineered.parquet"
WINDOW = 10         # Finestra mobile (minuti)
MAX_TIME_GAP = 300  # 5 minuti: se il buco è maggiore, spezza il viaggio

print(f"Carico dataset: {INPUT_FILE}")
df = pd.read_parquet(INPUT_FILE)

# Assicuriamoci che il tempo sia datetime
df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])
df = df.sort_values(["MMSI", "BaseDateTime"]).reset_index(drop=True)

# =================================================================
# 1. SEGMENTAZIONE VIAGGI (TRIP SPLITTING) - CRUCIALE PRIMA DEI CALCOLI
# =================================================================
print("Identificazione Trip_ID (Segmentazione temporale)...")

# Calcoliamo il Delta T (secondi) rispetto alla riga precedente
# fillna(0) serve per la prima riga assoluta
df["dt_sec"] = df.groupby("MMSI")["BaseDateTime"].diff().dt.total_seconds().fillna(0)

# Condizione Nuova Sequenza:
# 1. La nave cambia (MMSI diverso) - gestito dal groupby, ma il shift globale aiuta
# 2. C'è un buco temporale > MAX_TIME_GAP
condition = (df["MMSI"] != df["MMSI"].shift(1)) | (df["dt_sec"] > MAX_TIME_GAP)

# Creiamo un ID univoco globale per ogni segmento continuo
df["Trip_ID"] = condition.cumsum()

print(f"   Trovati {df['Trip_ID'].nunique()} segmenti continui di navigazione.")

# =================================================================
# 2. Derivate Spaziali (Calcolate per TRIP, non per MMSI)
# =================================================================
# Ora raggruppiamo per Trip_ID. Così se c'è un buco temporale, 
# il diff() non scavalca il buco (perché il Trip_ID cambia).
print("Calcolo derivate spaziali (Metri/min)...")

# dX, dY: Spostamento in metri
df["dX"] = df.groupby("Trip_ID")["X"].diff()
df["dY"] = df.groupby("Trip_ID")["Y"].diff()

# speed_xy: Velocità scalare (Ground Truth)
# Riempiamo i NaN iniziali di ogni trip con 0
df["speed_xy"] = np.sqrt(df["dX"]**2 + df["dY"]**2).fillna(0)

# =================================================================
# 3. Derivate Cinematiche (Segnale Radio)
# =================================================================
df["dSOG"] = df.groupby("Trip_ID")["SOG"].diff().fillna(0)
df["dCOG"] = df.groupby("Trip_ID")["COG"].diff().fillna(0)

# =================================================================
# 4. Statistiche Mobili (Rolling Features)
# =================================================================
features_to_roll = ["SOG", "COG", "speed_xy", "dSOG", "dCOG"]

print(f"Calcolo rolling stats (finestra {WINDOW} min)...")
for col in features_to_roll:
    # Usiamo Trip_ID anche qui! 
    # Così la media mobile non mischia dati di viaggi diversi.
    grouper = df.groupby("Trip_ID")[col]
    
    df[f"{col}_mean{WINDOW}"] = grouper.transform(lambda x: x.rolling(WINDOW, min_periods=1).mean())
    df[f"{col}_std{WINDOW}"] = grouper.transform(lambda x: x.rolling(WINDOW, min_periods=1).std())

# =================================================================
# 5. Pulizia e Salvataggio
# =================================================================
# Rimuoviamo colonne di servizio non necessarie per il training
df.drop(columns=["dt_sec"], inplace=True)

# I NaN rimanenti (i primissimi punti di ogni trip) vanno a 0
df = df.fillna(0)

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
df.to_parquet(OUTPUT_FILE, index=False)

print(f"Salvato: {OUTPUT_FILE}")
print(f"Righe finali: {len(df):,}")
print("   ✅ Calcoli fisici protetti da 'Trip Splitting'")