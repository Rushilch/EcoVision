"""
EcoVision - Flood Risk Data Ingestion & Preprocessing
----------------------------------------------------

Dataset type:
- Kaggle Flood Probability Prediction
- Risk-factor based (NOT rainfall/river driven)

Files present:
- train.csv  -> used for preprocessing & modeling
- test.csv   -> ignored here (used later for inference)

Target:
- FloodProbability

Author: EcoVision Project
"""

import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# ==============================
# PATH CONFIGURATION
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

TRAIN_FILE = "train.csv"
OUTPUT_FILE = "flood_clean.csv"

# ==============================
# EXPECTED COLUMNS (FROM TRAIN.CSV)
# ==============================

FEATURE_COLUMNS = [
    "PopulationScore",
    "WetlandLoss",
    "InadequatePlanning",
    "PoliticalFactors"
]

TARGET_COLUMN = "FloodProbability"

# ==============================
# HELPER FUNCTIONS
# ==============================

def load_train_data():
    path = os.path.join(RAW_DIR, TRAIN_FILE)
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ train.csv not found at {path}")

    print(f"📄 Using training file: {path}")
    return pd.read_csv(path)

# ==============================
# CORE PREPROCESSING
# ==============================

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ----------------------------
    # Validate required columns
    # ----------------------------
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"❌ Missing required columns: {missing}\n"
            f"Available columns: {df.columns.tolist()}"
        )

    # ----------------------------
    # Select relevant columns
    # ----------------------------
    df = df[required]

    # ----------------------------
    # Handle missing values
    # ----------------------------
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(df[FEATURE_COLUMNS].mean())
    df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna(df[TARGET_COLUMN].mean())

    # ----------------------------
    # Normalize features
    # ----------------------------
    scaler = MinMaxScaler()
    df[FEATURE_COLUMNS] = scaler.fit_transform(df[FEATURE_COLUMNS])

    # ----------------------------
    # Clamp target to [0,1]
    # ----------------------------
    df[TARGET_COLUMN] = df[TARGET_COLUMN].clip(0, 1)

    return df

# ==============================
# MAIN PIPELINE
# ==============================

def main():
    print("\n🌊 EcoVision – Flood Risk Preprocessing\n")

    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("📥 Loading flood training data...")
    df_raw = load_train_data()
    print("✔ Loaded successfully")
    print("Shape:", df_raw.shape)
    print("Columns:")
    print(df_raw.columns.tolist())

    print("\n🧠 Preprocessing flood features...")
    df_clean = preprocess(df_raw)
    print("✔ Preprocessing complete")

    print("\n📊 Sample processed data:")
    print(df_clean.head())

    output_path = os.path.join(PROCESSED_DIR, OUTPUT_FILE)
    df_clean.to_csv(output_path, index=False)

    print(f"\n💾 Saved processed flood dataset to:\n{output_path}")
    print("\n🎉 Flood preprocessing DONE\n")

# ==============================
# ENTRY POINT
# ==============================

if __name__ == "__main__":
    main()
