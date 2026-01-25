"""
AIS Data Explorer
-----------------
Scopo: analizzare un singolo file AIS per identificare
- copertura geografica (aree con più traffico)
- tipi di nave più rappresentati
- MMSI più attivi
"""

import pandas as pd
import matplotlib.pyplot as plt

# === CONFIG ===
FILE = "./input_raw/AIS_2024_01_01.csv"   # nome del file scaricato
SAMPLE_SIZE = 50000           # numero punti da campionare per la mappa
SHOW_MAP = True               # True = genera scatter geografico

# === 1. Caricamento rapido ===
cols = ['MMSI','BaseDateTime','LAT','LON','SOG','COG','Heading','VesselType']
df = pd.read_csv(FILE, usecols=[c for c in cols if c in pd.read_csv(FILE, nrows=0).columns],
                 low_memory=False)

print(f"Totale righe: {len(df):,}")
print("Colonne disponibili:", list(df.columns))

# === 2. Range geografico ===
lat_min, lat_max = df['LAT'].min(), df['LAT'].max()
lon_min, lon_max = df['LON'].min(), df['LON'].max()
print(f"\nRange coordinate: LAT {lat_min:.2f} → {lat_max:.2f}, LON {lon_min:.2f} → {lon_max:.2f}")

# === 3. Tipi nave (se colonna presente) ===
if 'VesselType' in df.columns:
    print("\nTop 10 tipi di nave:")
    print(df['VesselType'].value_counts().head(10))
else:
    print("\nColonna 'VesselType' non presente in questo file.")

# === 4. MMSI più attivi ===
print("\nTop 10 MMSI per numero di punti:")
print(df['MMSI'].value_counts().head(10))

# === 5. Campione e visualizzazione mappa ===
if SHOW_MAP:
    sample = df.sample(min(SAMPLE_SIZE, len(df)))
    plt.figure(figsize=(8,6))
    plt.scatter(sample['LON'], sample['LAT'], s=1, alpha=0.4)
    plt.title("Distribuzione geografica dei punti AIS (campione)")
    plt.xlabel("Longitudine"); plt.ylabel("Latitudine")
    plt.show()

"""
Interpretazione:
----------------
1. Il grafico scatter mostra dove sono concentrate le navi → scegli una zona con alta densità.
2. I limiti LAT/LON ottenuti ti servono per impostare LAT_MIN, LAT_MAX, LON_MIN, LON_MAX nello script di filtraggio.
3. 'VesselType' ti indica le categorie più frequenti → scegli una (es. Cargo o Tanker).
4. Gli MMSI più attivi possono essere usati per testare modelli singoli.
"""
