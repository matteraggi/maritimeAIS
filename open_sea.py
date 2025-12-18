"""
open_sea.py
-----------
Input:  ./preprocessed/ais_feature_engineered.parquet (Dati con fisica calcolata)
Output: ./preprocessed/ais_final.parquet (Solo navigazione stabile)

Logica:
1. Filtra le righe con SOG < Soglia.
2. RI-CALCOLA i Trip_ID. Se il filtro ha creato un buco nel mezzo di un viaggio 
   (es. la nave ha rallentato per 1 ora e poi ripreso), quel viaggio viene diviso in due.
"""

import pandas as pd
import numpy as np
import os

# === CONFIGURAZIONE ===
INPUT_FILE = "./preprocessed/ais_feature_engineered.parquet"
OUTPUT_FILE = "./preprocessed/ais_final.parquet"

# Soglia SOG (z-score normalizzato o nodi, dipende dai tuoi dati).
# Se i dati sono standardizzati (mean=0, std=1), 0.2 è una buona soglia conservativa.
SOG_THRESHOLD = 0.2 
MAX_TIME_GAP = 600 # 10 minuti: se il filtro crea un buco > 10 min, spezza il trip

def apply_open_sea_filter():
    print(f"🌊 AVVIO FILTRO OPEN SEA (SOG > {SOG_THRESHOLD})")
    
    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"Input non trovato: {INPUT_FILE}")

    # 1. Carica dataset con feature fisiche già pronte
    df = pd.read_parquet(INPUT_FILE)
    initial_len = len(df)
    print(f"   Righe iniziali: {initial_len}")

    # 2. Applica il Filtro Velocità
    # (Assumiamo che la colonna si chiami 'SOG')
    df_fast = df[df["SOG"] > SOG_THRESHOLD].copy()
    
    # 3. RICALCOLO TRIP ID (Repair Strategy)
    # Il filtro potrebbe aver creato dei buchi. Dobbiamo sanarli.
    print("   🔧 Riparazione e ricalcolo Trip IDs...")
    
    df_fast = df_fast.sort_values(["MMSI", "BaseDateTime"])
    
    # Calcoliamo il tempo tra una riga e l'altra (dopo aver rimosso le lente)
    # Se abbiamo rimosso delle righe intermedie, dt_sec sarà grande.
    df_fast["dt_sec"] = df_fast.groupby("MMSI")["BaseDateTime"].diff().dt.total_seconds().fillna(0)
    
    # Nuova logica di rottura viaggio:
    # 1. Cambio nave (MMSI)
    # 2. Buco temporale > MAX_TIME_GAP (dovuto al filtro o alla perdita di segnale)
    # 3. (Opzionale) Cambio drastico del vecchio Trip_ID originale
    condition = (df_fast["MMSI"] != df_fast["MMSI"].shift(1)) | \
                (df_fast["dt_sec"] > MAX_TIME_GAP)
                
    df_fast["Trip_ID"] = condition.cumsum()
    
    # 4. Rimuovi viaggi troppo corti (Short Trip Removal)
    # Se un "viaggio" è rimasto con solo 5 punti, non serve all'AI.
    trip_counts = df_fast["Trip_ID"].value_counts()
    min_sequence_len = 35 # Un po' più della seq_len del modello (30)
    valid_trips = trip_counts[trip_counts >= min_sequence_len].index
    
    df_final = df_fast[df_fast["Trip_ID"].isin(valid_trips)].copy()
    
    # Pulizia colonne temporanee
    df_final.drop(columns=["dt_sec"], inplace=True)
    
    # 5. Statistiche Finali
    final_len = len(df_final)
    kept_pct = (final_len / initial_len) * 100
    
    print(f"\n   ✅ FILTRO COMPLETATO")
    print(f"   Righe rimaste: {final_len} ({kept_pct:.1f}%)")
    print(f"   Viaggi Validi (Trip_ID): {df_final['Trip_ID'].nunique()}")
    print(f"   Scartati {len(trip_counts) - len(valid_trips)} segmenti troppo corti (<{min_sequence_len} step).")

    # 6. Salva
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_final.to_parquet(OUTPUT_FILE)
    print(f"   💾 Dataset salvato: {OUTPUT_FILE}")

if __name__ == "__main__":
    apply_open_sea_filter()