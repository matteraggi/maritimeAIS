"""
check_preprocessing.py
----------------------
Scopo:
  - Controllare la qualità del file pre-processato
  - Verificare che i dati siano numericamente coerenti, completi e regolari
  - Visualizzare controlli base su tempo, posizione e fisica

Input:  ./preprocessed/ais_preprocessed.parquet
Output: report testuale e grafico di distribuzione
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# === CONFIGURAZIONE ===
FILE_PATH = "./preprocessed/ais_preprocessed.parquet"

# === 1. Controllo esistenza ===
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"File non trovato: {FILE_PATH}")

# === 2. Caricamento ===
print(f"Caricamento file: {FILE_PATH}")
df = pd.read_parquet(FILE_PATH)
print(f"Righe totali: {len(df):,}")
print(f"Colonne: {list(df.columns)}\n")

# === 3. Controllo NaN e infiniti ===
print("=== Valori mancanti ===")
print(df.isna().sum(), "\n")

# === 4. Statistiche base ===
print("=== Statistiche sintetiche ===")
print(df.describe().round(3), "\n")

# === 5. Delta temporale tra punti successivi ===
df["BaseDateTime"] = pd.to_datetime(df["BaseDateTime"])
df["delta_t"] = df.groupby("MMSI")["BaseDateTime"].diff().dt.total_seconds()
print("=== Distribuzione intervalli temporali (sec) ===")
print(df["delta_t"].describe().round(3))
plt.hist(df["delta_t"].dropna(), bins=40)
plt.title("Distribuzione intervalli temporali (sec)")
plt.xlabel("Δt (secondi)")
plt.ylabel("Conteggio")
plt.show()

# === 6. Controllo coerenza geografica ===
sample = df.sample(min(5000, len(df)))
plt.scatter(sample["X"], sample["Y"], s=2, alpha=0.5)
plt.title("Distribuzione spaziale (X,Y) – campione")
plt.xlabel("X (metri normalizzati)")
plt.ylabel("Y (metri normalizzati)")
plt.show()

# === 7. Controllo fisico su SOG ===
print("=== Controllo velocità (SOG normalizzata) ===")
print(f"Media: {df['SOG'].mean():.3f}, Dev std: {df['SOG'].std():.3f}")
plt.hist(df["SOG"], bins=50)
plt.title("Distribuzione SOG (normalizzata)")
plt.xlabel("SOG (z-score)")
plt.ylabel("Conteggio")
plt.show()

# === 8. Controllo MMSI ===
print("=== Numero di navi distinte ===")
print(df["MMSI"].nunique())
print("Punti per MMSI:")
print(df["MMSI"].value_counts().head(10))
