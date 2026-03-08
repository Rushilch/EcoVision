
# 🌍 EcoVision  
### AI-Powered Environmental Risk Prediction & Sustainability Dashboard

EcoVision is a production-grade academic prototype that predicts **Urban Heat Island (UHI)** intensity, **Flood Risk**, and **Plastic Waste Hotspots**, and provides **explainable, actionable insights** through an interactive dashboard.

The system is fully implemented in **Python**, follows a **modular layered architecture**, and is suitable for **live demonstration and viva defense**.

---

## 🚀 Features

- 🌡️ Urban Heat Island prediction (area heatmap)
- 🌊 Flood risk probability estimation
- ♻️ Plastic waste hotspot detection & route optimization
- 🧠 Explainable AI using **SHAP** and **LIME**
- 🔌 REST API using **FastAPI**
- 📊 Interactive dashboard using **Streamlit**
- 📍 Location pinning with persistent predictions
- 🚛 Optimized plastic waste collection routes

---


## 🧱 System Architecture

EcoVision follows a **layered, decoupled architecture**:


```

Data Layer  
↓  
Preprocessing Layer  
↓  
Machine Learning Layer  
↓  
Explainable AI Layer  
↓  
Backend API Layer (FastAPI)  
↓  
Dashboard Layer (Streamlit)

```

Each layer is independent, testable, and reusable.

---
---


## 📊 Data

- https://www.kaggle.com/datasets/prajwaldongre/global-plastic-waste-2023-a-country-wise-analysis
- https://www.kaggle.com/code/devraai/urban-heat-island-analysis-prediction/input
- https://www.kaggle.com/datasets/naiyakhalid/flood-prediction-dataset
- https://www.kaggle.com/datasets/pratyushpuri/urban-flood-risk-data-global-city-analysis-2025

---

## 📁 Project Structure


```

ecovision/  
│  
├── data/  
│ ├── raw/ # Original datasets  
│ ├── processed/ # Cleaned datasets  
│ └── simulated/ # Generated proxy data  
│  
├── preprocessing/  
│ ├── ingest_uhi.py  
│ ├── ingest_flood.py  
│ └── ingest_plastic.py  
│  
├── models/  
│ ├── uhi_model.py  
│ ├── flood_model.py  
│ └── plastic_model.py  
│  
├── explainability/  
│ ├── shap_explainer.py  
│ └── lime_explainer.py  
│  
├── backend/  
│ ├── api.py # FastAPI server  
│ ├── database.py # SQLite persistence  
│ └── schemas.py  
│  
├── utils/  
│ ├── geo_utils.py  
│ ├── recommendations.py  
│ └── plastic_route_optimization.py  
│  
├── dashboard/  
│ └── app.py # Streamlit dashboard  
│  
├── req.txt  
└── README.md
└── runecovision.py

```

---

## ⚙️ Requirements

- Python **3.10+**
- pip
- Virtual environment (recommended)

---

## 📦 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/rushilch/ecovision.git
cd ecovision

```

----------

### 2️⃣ Create & Activate Virtual Environment

**Windows**

```bash
python -m venv venv
venv\Scripts\activate

```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate

```

----------

### 3️⃣ Install Dependencies

```bash
pip install -r req.txt

```

If missing:

```bash
pip install streamlit fastapi uvicorn shap lime folium streamlit-folium geopy

```

----------

## 🗂️ Data Setup

Place datasets in:

```
data/raw/

```

Supported formats:

-   `.csv`
    
-   `.xlsx`
    

If real data is unavailable, the system uses **statistically realistic simulated data**.

----------

## 🔄 Data Preprocessing

Run preprocessing scripts **once**:

```bash
python preprocessing/ingest_uhi.py
python preprocessing/ingest_flood.py
python preprocessing/ingest_plastic.py

```

This generates cleaned datasets in:

```
data/processed/

```

----------

## 🤖 Model Training

Models are trained.

You can retrain manually if needed:

```bash
python training/train_uhi_model.py
python training/train_flood_model.py
python training/train_plastic_model.py

```

----------

## 🔌 Running

run_ecovision.py starts the backend and runs the server dashboard

```
python run_ecovision.py

```

Dashboard opens at:

```
http://localhost:8501

```

----------

## 🧭 Dashboard Usage

### Sidebar Controls

-   Select **City / Region**
    
-   Choose **Time Range**
    
-   Switch between modules
    

### Tabs

1.  **Risk Maps**
    
    -   UHI heatmap
        
    -   Flood distribution
        
    -   Plastic waste world map
        
2.  **Predict by Inputs**
    
    -   Manual feature input
        
3.  **Plastic Collection Planner**
    
    -   Pin waste hotspots
        
    -   Generate optimized routes
        

----------

## 🧠 Explainable AI

-   **SHAP**: Global & local feature importance
    
-   **LIME**: Instance-level explanations
    
-   Available via API and dashboard views
    

    
