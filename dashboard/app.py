"""
EcoVision - Multi-Page Streamlit Dashboard (Stable + Interactive)
----------------------------------------------------------------
Features:
- Overview maps
- API-driven predictions
- Pin a location on map
- Visible marker on pinned point
- Clear user instructions

Author: EcoVision Project
"""

import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from streamlit_plotly_events import plotly_events

# ==============================
# CONFIG
# ==============================

API_BASE = "http://127.0.0.1:8000"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")

st.set_page_config(
    page_title="EcoVision Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 EcoVision – Environmental Risk & Explainable AI")
st.markdown(
    "AI-powered **Urban Heat**, **Flood Risk**, and **Plastic Waste** intelligence."
)
st.markdown("---")

# ==============================
# LOAD DATA
# ==============================

@st.cache_data
def load_data():
    uhi = pd.read_csv(os.path.join(DATA_PROCESSED, "uhi_clean.csv"))
    flood = pd.read_csv(os.path.join(DATA_PROCESSED, "flood_clean.csv"))
    plastic = pd.read_csv(os.path.join(DATA_RAW, "plastic.csv"))
    return uhi, flood, plastic

uhi_df, flood_df, plastic_raw = load_data()

# ==============================
# SESSION STATE
# ==============================

if "pinned_location" not in st.session_state:
    st.session_state.pinned_location = None

# ==============================
# TABS
# ==============================

tab1, tab2, tab3 = st.tabs(
    ["🗺️ Overview Maps", "🧮 Predict by Inputs", "📍 Pin a Location"]
)

# ======================================================
# TAB 1: OVERVIEW MAPS
# ======================================================

with tab1:
    st.header("🗺️ Environmental Risk Overview")

    st.subheader("🌡️ Urban Heat Island Map")
    fig = px.scatter_map(
        uhi_df,
        lat="Latitude",
        lon="Longitude",
        color="UHI",
        size="UHI",
        color_continuous_scale="hot",
        zoom=4,
        height=450
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("🌊 Flood Risk Distribution")
    fig = px.histogram(
        flood_df,
        x="FloodProbability",
        nbins=30
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("♻️ Global Plastic Waste (Overview)")
    st.dataframe(plastic_raw.head(10))

# ======================================================
# TAB 2: PREDICT VIA INPUTS
# ======================================================

with tab2:
    st.header("🧮 Predict Environmental Risk (Manual Inputs)")

    with st.form("uhi_form"):
        LST = st.slider("Land Surface Temperature (normalized)", 0.0, 1.0, 0.6)
        NDVI = st.slider("NDVI", 0.0, 1.0, 0.3)
        NDBI = st.slider("NDBI", 0.0, 1.0, 0.7)
        Albedo = st.slider("Albedo", 0.0, 1.0, 0.4)
        LULC = st.selectbox("Land Use Class", [0, 1, 2, 3])

        submit = st.form_submit_button("Predict UHI")

    if submit:
        res = requests.post(
            f"{API_BASE}/predict/uhi",
            json={
                "LST": LST,
                "NDVI": NDVI,
                "NDBI": NDBI,
                "Albedo": Albedo,
                "LULC": LULC
            }
        )

        if res.ok:
            data = res.json()
            st.success(f"🌡️ UHI Score: {data['UHI']}")
            st.info(f"Zone: {data['Zone']}")

# ======================================================
# TAB 3: PIN A LOCATION (MULTI-POINT + COUNTRY STORAGE)
# ======================================================

from streamlit_folium import st_folium
import folium
from geopy.geocoders import Nominatim

with tab3:
    st.header("📍 Pin Locations & Track Predictions")

    st.info(
        "🧭 **How to pin locations:**\n\n"
        "1. Zoom or pan the map\n"
        "2. Click anywhere to pin a location\n"
        "3. Press **Predict for this Location**\n"
        "4. All previous points will remain visible\n"
    )

    # ------------------------------
    # Session state initialization
    # ------------------------------

    if "pinned_points" not in st.session_state:
        st.session_state.pinned_points = []

    if "current_click" not in st.session_state:
        st.session_state.current_click = None

    geolocator = Nominatim(user_agent="ecovision_app")

    # ------------------------------
    # Base map
    # ------------------------------

    m = folium.Map(
        location=[
            uhi_df["Latitude"].mean(),
            uhi_df["Longitude"].mean()
        ],
        zoom_start=5,
        tiles="OpenStreetMap"
    )

    # Add markers for all stored points
    for p in st.session_state.pinned_points:
        folium.Marker(
            [p["lat"], p["lon"]],
            popup=f"{p['country']} | UHI: {p['uhi']}",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)

    # Add marker for current click (red)
    if st.session_state.current_click:
        folium.Marker(
            [
                st.session_state.current_click["lat"],
                st.session_state.current_click["lon"]
            ],
            popup="Current Selection",
            icon=folium.Icon(color="red", icon="map-marker")
        ).add_to(m)

    map_data = st_folium(
        m,
        height=500,
        width="100%"
    )

    # ------------------------------
    # Capture click
    # ------------------------------

    if map_data and map_data.get("last_clicked"):
        st.session_state.current_click = {
            "lat": map_data["last_clicked"]["lat"],
            "lon": map_data["last_clicked"]["lng"]
        }

    # ------------------------------
    # Prediction logic
    # ------------------------------

    if st.session_state.current_click:
        lat = st.session_state.current_click["lat"]
        lon = st.session_state.current_click["lon"]

        st.success(f"📌 Selected: Latitude **{lat:.4f}**, Longitude **{lon:.4f}**")

        if st.button("Predict for this Location"):

            # Reverse geocode country
            try:
                location = geolocator.reverse((lat, lon), language="en")
                country = location.raw["address"].get("country", "Unknown")
            except Exception:
                country = "Unknown"

            with st.spinner("Calling EcoVision APIs..."):

                uhi = requests.post(
                    f"{API_BASE}/predict/uhi",
                    json={
                        "LST": abs(lat % 1),
                        "NDVI": 1 - abs(lat % 1),
                        "NDBI": abs(lat % 1),
                        "Albedo": 0.4,
                        "LULC": 2
                    }
                ).json()

                flood = requests.post(
                    f"{API_BASE}/predict/flood",
                    json={
                        "PopulationScore": min(abs(lat) / 90, 1),
                        "WetlandLoss": min(abs(lon) / 180, 1),
                        "InadequatePlanning": 0.6,
                        "PoliticalFactors": 0.5
                    }
                ).json()

                plastic = requests.post(
                    f"{API_BASE}/predict/plastic",
                    json={
                        "Country": 10,
                        "Main_Sources": 2,
                        "Coastal_Waste_Risk": 1,
                        "Total_Plastic_Waste_MT": 0.6,
                        "Per_Capita_Waste_KG": 0.5,
                        "Recycling_Rate": 0.4
                    }
                ).json()

            record = {
                "lat": lat,
                "lon": lon,
                "country": country,
                "uhi": uhi["UHI"],
                "flood": flood["FloodProbability"],
                "plastic": plastic["PlasticWasteRisk"]
            }

            st.session_state.pinned_points.append(record)
            st.session_state.current_click = None

            st.success(f"✅ Prediction stored for **{country}**")

    else:
        st.warning("Click on the map to select a location")

    # ------------------------------
    # Display stored predictions
    # ------------------------------

    if st.session_state.pinned_points:
        st.subheader("📊 Stored Location Predictions")

        df_points = pd.DataFrame(st.session_state.pinned_points)
        st.dataframe(df_points, width="stretch")

        if st.button("Clear All Stored Points"):
            st.session_state.pinned_points = []
            st.experimental_rerun()



# ==============================
# FOOTER
# ==============================

st.markdown("---")
st.markdown(
    "<center><b>EcoVision</b> | Interactive Location-Based Environmental Intelligence</center>",
    unsafe_allow_html=True
)
