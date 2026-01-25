"""
Parquet Viewer Script
---------------------
Scopo:
  - Leggere e visualizzare rapidamente un file .parquet
  - Mostrare struttura, dimensioni e prime righe ordinate

Uso:
  python view_parquet.py
"""

import pandas as pd
import os

# === CONFIGURAZIONE ===
FILE_PATH = "./preprocessed/ais_preprocessed.parquet"  # cambia se serve
N_ROWS = 20                                            # quante righe mostrare

# === 1. Controllo esistenza file ===
if not os.path.exists(FILE_PATH):
    raise FileNotFoundError(f"File non trovato: {FILE_PATH}")

# === 2. Caricamento ===
print(f"Caricamento file: {FILE_PATH}")
df = pd.read_parquet(FILE_PATH)
print(f"Righe totali: {len(df):,}")
print(f"Colonne: {list(df.columns)}\n")

# === 3. Info strutturali ===
print("=== Info dataset ===")
print(df.info(memory_usage='deep'))
print("\n=== Statistiche sintetiche ===")
print(df.describe().round(3))
print("\n")

# === 4. Visualizzazione righe iniziali ordinate ===
cols_sort = [c for c in ['MMSI', 'BaseDateTime'] if c in df.columns]
if cols_sort:
    df = df.sort_values(cols_sort)
print(f"=== Prime {N_ROWS} righe ===")
print(df.head(N_ROWS))

# === 5. (Opzionale) esporta un'anteprima CSV leggibile ===
df.head(1000).to_csv("preview_sample.csv", index=False)
print("\nAnteprima salvata in: preview_sample.csv (prime 1000 righe)")
