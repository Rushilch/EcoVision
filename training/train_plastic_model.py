"""
EcoVision - Plastic Waste Risk Model Training
--------------------------------------------

This script:
1. Loads preprocessed plastic waste data
2. Splits train/test sets
3. Trains a Random Forest Regressor
4. Evaluates model performance
5. Saves trained model for inference & explainability

Author: EcoVision Project
"""

import os
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ==============================
# PATH CONFIGURATION
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "plastic_clean.csv")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "saved")

MODEL_FILE = "plastic_random_forest.pkl"

# ==============================
# MODEL CONFIGURATION
# ==============================

FEATURES = [
    "Country",
    "Main_Sources",
    "Coastal_Waste_Risk",
    "Total_Plastic_Waste_MT",
    "Per_Capita_Waste_KG",
    "Recycling_Rate"
]

TARGET = "PlasticWasteRisk"

RANDOM_STATE = 42

# ==============================
# HELPER FUNCTIONS
# ==============================

def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"❌ Processed plastic data not found at {DATA_PATH}"
        )
    return pd.read_csv(DATA_PATH)

def train_model(X_train, y_train):
    model = RandomForestRegressor(
        n_estimators=400,
        max_depth=14,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print("\n📊 PLASTIC MODEL PERFORMANCE")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"R²   : {r2:.4f}")

def save_model(model):
    os.makedirs(MODEL_DIR, exist_ok=True)
    model_path = os.path.join(MODEL_DIR, MODEL_FILE)
    joblib.dump(model, model_path)
    print(f"\n💾 Model saved to: {model_path}")

# ==============================
# MAIN PIPELINE
# ==============================

def main():
    print("\n♻️ EcoVision – Plastic Waste Model Training\n")

    print("📥 Loading processed plastic data...")
    df = load_data()
    print("✔ Data loaded")
    print("Shape:", df.shape)

    print("\n📋 Using features:")
    for f in FEATURES:
        print(" -", f)

    X = df[FEATURES]
    y = df[TARGET]

    print("\n🔀 Splitting train/test data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE
    )
    print("✔ Split complete")

    print("\n🤖 Training Random Forest plastic waste model...")
    model = train_model(X_train, y_train)
    print("✔ Training complete")

    print("\n🧪 Evaluating model...")
    evaluate_model(model, X_test, y_test)

    print("\n💾 Saving trained model...")
    save_model(model)

    print("\n🎉 Plastic waste model training COMPLETE\n")

# ==============================
# ENTRY POINT
# ==============================

if __name__ == "__main__":
    main()
