"""
EcoVision - Urban Heat Island Data Ingestion & Feature Engineering
----------------------------------------------------------------

This script:
1. Loads Kaggle UHI dataset (city-level)
2. Derives UHI-relevant features
3. Converts proxy variables to EcoVision standard
4. Computes UHI intensity score
5. Normalizes features
6. Saves clean dataset for modeling

Dataset reality:
- No satellite NDVI/NDBI → derived from urban proxies
- Scientifically valid for academic use

Author: EcoVision Project
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# ==============================
# PATH SETUP (WINDOWS SAFE)
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

OUTPUT_FILE = "uhi_clean.csv"

# ==============================
# FEATURE DEFINITIONS
# ==============================

FINAL_FEATURES = [
    "LST",        # Land Surface Temperature (proxy)
    "NDVI",       # Greenness proxy
    "NDBI",       # Built-up proxy
    "Albedo",     # Derived surface reflectance
    "LULC",       # Land use encoding
    "Latitude",
    "Longitude",
    "UHI"
]

# ==============================
# HELPER FUNCTIONS
# ==============================

def find_uhi_file():
    for f in os.listdir(RAW_DIR):
        if "urban_heat" in f.lower():
            return os.path.join(RAW_DIR, f)
    raise FileNotFoundError("❌ Urban Heat Island dataset not found in data/raw")

def load_data():
    path = find_uhi_file()
    print(f"📄 Using dataset: {path}")

    if path.endswith(".xlsx"):
        return pd.read_excel(path)
    return pd.read_csv(path)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert Kaggle UHI dataset into EcoVision-compatible features
    """

    df = df.copy()

    # ----------------------------
    # Rename base columns
    # ----------------------------
    df.rename(columns={
        "Temperature (°C)": "LST",
        "Urban Greenness Ratio (%)": "Greenness",
        "Population Density (people/km²)": "PopDensity",
        "Energy Consumption (kWh)": "Energy",
        "Land Cover": "LandCover"
    }, inplace=True)

    # ----------------------------
    # Feature engineering
    # ----------------------------

    # NDVI proxy (normalize greenness)
    df["NDVI"] = df["Greenness"] / 100

    # NDBI proxy (urban intensity)
    df["NDBI"] = MinMaxScaler().fit_transform(df[["PopDensity"]])

    # Albedo proxy (inverse greenness)
    df["Albedo"] = 1 - df["NDVI"]

    # Encode land cover
    df["LULC"] = df["LandCover"].astype("category").cat.codes

    # ----------------------------
    # UHI intensity computation
    # ----------------------------
    df["UHI"] = (
        0.45 * MinMaxScaler().fit_transform(df[["LST"]]).flatten() +
        0.25 * df["NDBI"] -
        0.20 * df["NDVI"] +
        0.10 * MinMaxScaler().fit_transform(df[["Energy"]]).flatten()
    ).clip(0, 1)

    # ----------------------------
    # Select final columns
    # ----------------------------
    df_final = df[[
        "LST",
        "NDVI",
        "NDBI",
        "Albedo",
        "LULC",
        "Latitude",
        "Longitude",
        "UHI"
    ]]

    # Handle missing values
    df_final = df_final.fillna(df_final.mean())

    # Normalize numeric features (except lat/long)
    scaler = MinMaxScaler()
    scale_cols = ["LST", "NDVI", "NDBI", "Albedo"]
    df_final[scale_cols] = scaler.fit_transform(df_final[scale_cols])

    return df_final

# ==============================
# MAIN PIPELINE
# ==============================

def main():
    print("\n🚀 EcoVision – Urban Heat Island Feature Pipeline\n")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("📥 Loading raw dataset...")
    df_raw = load_data()
    print("✔ Loaded successfully")
    print("Columns found:")
    print(df_raw.columns.tolist())

    print("\n🧠 Engineering UHI features...")
    df_clean = preprocess(df_raw)

    print("✔ Feature engineering completed")
    print("\n📊 Sample processed data:")
    print(df_clean.head())

    output_path = os.path.join(PROCESSED_DIR, OUTPUT_FILE)
    df_clean.to_csv(output_path, index=False)

    print(f"\n💾 Saved processed dataset to:\n{output_path}")
    print("\n🎉 UHI preprocessing DONE\n")

# ==============================
# ENTRY POINT
# ==============================

if __name__ == "__main__":
    main()
