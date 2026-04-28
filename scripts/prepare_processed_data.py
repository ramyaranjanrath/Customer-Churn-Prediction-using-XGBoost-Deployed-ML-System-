import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),'..')))

from src.data.preprocess import preprocess_data
from src.features.build_features import build_features

RAW = "data/raw/data.csv"
OUT = "data/processed/processed.csv"

df = pd.read_csv(RAW)
df = preprocess_data(df, target_col="Churn")

if "Churn" in df.columns and df["Churn"].dtype == "object":
    df["Churn"] = df["Churn"].str.strip().map({"No": 0, "Yes": 1}).astype("Int64")

df_processed = build_features(df, target_col="Churn")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
df_processed.to_csv(OUT, index=False)

print(f"Processed dataset saved to {OUT}, Shape: {df_processed.shape}")