"""
EcoVision - Unified FastAPI Backend
----------------------------------

Provides:
- Prediction APIs for UHI, Flood, Plastic
- SHAP explainability (global + local)
- Health check endpoint

Author: EcoVision Project
"""

import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from explainability.shap_explainer import SHAPExplainer

# ==============================
# PATH CONFIGURATION
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "saved")
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "processed")

# ==============================
# LOAD MODELS
# ==============================

uhi_model = joblib.load(os.path.join(MODEL_DIR, "uhi_random_forest.pkl"))
flood_model = joblib.load(os.path.join(MODEL_DIR, "flood_random_forest.pkl"))
plastic_model = joblib.load(os.path.join(MODEL_DIR, "plastic_random_forest.pkl"))

# ==============================
# FASTAPI APP
# ==============================

app = FastAPI(
    title="EcoVision API",
    description="AI-Powered Environmental Risk Prediction & Explainability",
    version="1.0"
)

# ==============================
# REQUEST SCHEMAS
# ==============================

class UHIRequest(BaseModel):
    LST: float
    NDVI: float
    NDBI: float
    Albedo: float
    LULC: int

class FloodRequest(BaseModel):
    PopulationScore: float
    WetlandLoss: float
    InadequatePlanning: float
    PoliticalFactors: float

class PlasticRequest(BaseModel):
    Country: int
    Main_Sources: int
    Coastal_Waste_Risk: int
    Total_Plastic_Waste_MT: float
    Per_Capita_Waste_KG: float
    Recycling_Rate: float

# ==============================
# HEALTH CHECK
# ==============================

@app.get("/health")
def health():
    return {"status": "EcoVision API running"}

# ==============================
# PREDICTION ENDPOINTS
# ==============================

@app.post("/predict/uhi")
def predict_uhi(data: UHIRequest):
    try:
        X = pd.DataFrame([data.dict()])
        pred = float(uhi_model.predict(X)[0])
        return {
            "UHI": round(pred, 4),
            "Zone": "High" if pred > 0.6 else "Medium" if pred > 0.3 else "Low"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/flood")
def predict_flood(data: FloodRequest):
    try:
        X = pd.DataFrame([data.dict()])
        pred = float(flood_model.predict(X)[0])
        return {
            "FloodProbability": round(pred, 4),
            "RiskLevel": "High" if pred > 0.6 else "Medium" if pred > 0.3 else "Low"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/plastic")
def predict_plastic(data: PlasticRequest):
    try:
        X = pd.DataFrame([data.dict()])
        pred = float(plastic_model.predict(X)[0])
        return {
            "PlasticWasteRisk": round(pred, 4),
            "RiskLevel": "High" if pred > 0.6 else "Medium" if pred > 0.3 else "Low"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==============================
# SHAP EXPLAINABILITY ENDPOINT
# ==============================

@app.get("/explain/shap/{module}")
def explain_shap(module: str, index: int = 0):
    try:
        if module == "uhi":
            explainer = SHAPExplainer(
                "uhi_random_forest.pkl",
                "uhi_clean.csv",
                ["LST", "NDVI", "NDBI", "Albedo", "LULC"]
            )
        elif module == "flood":
            explainer = SHAPExplainer(
                "flood_random_forest.pkl",
                "flood_clean.csv",
                ["PopulationScore", "WetlandLoss", "InadequatePlanning", "PoliticalFactors"]
            )
        elif module == "plastic":
            explainer = SHAPExplainer(
                "plastic_random_forest.pkl",
                "plastic_clean.csv",
                [
                    "Country",
                    "Main_Sources",
                    "Coastal_Waste_Risk",
                    "Total_Plastic_Waste_MT",
                    "Per_Capita_Waste_KG",
                    "Recycling_Rate"
                ]
            )
        else:
            raise HTTPException(status_code=400, detail="Invalid module name")

        return {
            "global_summary": explainer.global_text_summary(),
            "local_summary": explainer.local_text_summary(index=index)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
