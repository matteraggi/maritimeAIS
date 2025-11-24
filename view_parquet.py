import pandas as pd

FILE_PATH = "./preprocessed/ais_final.parquet"

df = pd.read_parquet(FILE_PATH)

print("=== COLONNE ===")
print(df.columns.tolist())

print("\n=== INFO ===")
print(df.info())

print("\n=== PRIME RIGHE ===")
print(df.head())

print("\n=== STATISTICHE ===")
print(df.describe(include='all'))


for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype.name == 'category':
        print(f"{col} → valori unici:", df[col].unique()[:20])


df = df.sort_values("BaseDateTime")
df["delta"] = df["BaseDateTime"].diff().dt.total_seconds()
print(df["delta"].describe())
