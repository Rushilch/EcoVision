"""
EcoVision - Plastic Waste Data Ingestion & Preprocessing
-------------------------------------------------------

Dataset:
- Country-level plastic waste dataset (Kaggle-style)

This script:
1. Loads plastic.csv
2. Encodes categorical variables
3. Normalizes numeric features
4. Computes PlasticWasteRisk score
5. Saves clean dataset for ML modeling

Author: EcoVision Project
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# ==============================
# PATH CONFIGURATION
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

RAW_FILE = "plastic.csv"
OUTPUT_FILE = "plastic_clean.csv"

# ==============================
# HELPER FUNCTIONS
# ==============================

def load_data():
    path = os.path.join(RAW_DIR, RAW_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"❌ plastic.csv not found at {path}"
        )
    return pd.read_csv(path)

# ==============================
# CORE PREPROCESSING
# ==============================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ----------------------------
    # Encode categorical columns
    # ----------------------------
    df["Main_Sources"] = LabelEncoder().fit_transform(df["Main_Sources"])
    df["Coastal_Waste_Risk"] = LabelEncoder().fit_transform(df["Coastal_Waste_Risk"])
    df["Country"] = LabelEncoder().fit_transform(df["Country"])

    # ----------------------------
    # Handle missing values
    # ----------------------------
    df = df.fillna(df.mean(numeric_only=True))

    # ----------------------------
    # Normalize numeric features
    # ----------------------------
    numeric_cols = [
        "Total_Plastic_Waste_MT",
        "Recycling_Rate",
        "Per_Capita_Waste_KG"
    ]

    scaler = MinMaxScaler()
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

    # ----------------------------
    # Plastic Waste Risk Score
    # ----------------------------
    df["PlasticWasteRisk"] = (
        0.45 * df["Total_Plastic_Waste_MT"] +
        0.30 * df["Per_Capita_Waste_KG"] -
        0.25 * df["Recycling_Rate"]
    ).clip(0, 1)

    # ----------------------------
    # Final column selection
    # ----------------------------
    df_final = df[
        [
            "Country",
            "Main_Sources",
            "Coastal_Waste_Risk",
            "Total_Plastic_Waste_MT",
            "Per_Capita_Waste_KG",
            "Recycling_Rate",
            "PlasticWasteRisk"
        ]
    ]

    return df_final

# ==============================
# MAIN PIPELINE
# ==============================

def main():
    print("\n♻️ EcoVision – Plastic Waste Preprocessing\n")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("📥 Loading plastic waste dataset...")
    df_raw = load_data()
    print("✔ Loaded successfully")
    print("Columns:")
    print(df_raw.columns.tolist())

    print("\n🧠 Preprocessing plastic waste data...")
    df_clean = preprocess(df_raw)
    print("✔ Preprocessing complete")

    print("\n📊 Sample processed data:")
    print(df_clean.head())

    output_path = os.path.join(PROCESSED_DIR, OUTPUT_FILE)
    df_clean.to_csv(output_path, index=False)

    print(f"\n💾 Saved processed plastic data to:\n{output_path}")
    print("\n🎉 Plastic preprocessing DONE\n")

# ==============================
# ENTRY POINT
# ==============================

if __name__ == "__main__":
    main()
