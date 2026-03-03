"""
EcoVision Dashboard – FINAL (INLINE GROQ, FIXED UHI MAP)
------------------------------------------------------
• UHI / Flood / Plastic visualization
• All Regions option
• Map-based pinning
• Country detection
• Predictions via FastAPI
• Rule-based explanations
• Groq AI suggestions (inline, no imports)
"""

import os
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
import folium
from streamlit_folium import st_folium
from geopy.geocoders import Nominatim
from dotenv import load_dotenv
from groq import Groq

# ==============================
# ENV + GROQ SETUP
# ==============================

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None

# ==============================
# CONFIG
# ==============================

API_BASE = "http://127.0.0.1:8000"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))

DATA_PROCESSED = os.path.join(PROJECT_ROOT, "data", "processed")
DATA_RAW = os.path.join(PROJECT_ROOT, "data", "raw")

st.set_page_config(
    page_title="EcoVision",
    page_icon="🌍",
    layout="wide"
)

# ==============================
# HELPERS
# ==============================

@st.cache_data
def load_data():
    uhi = pd.read_csv(os.path.join(DATA_PROCESSED, "uhi_clean.csv"))
    flood = pd.read_csv(os.path.join(DATA_PROCESSED, "flood_clean.csv"))
    plastic = pd.read_csv(os.path.join(DATA_RAW, "plastic.csv"))
    iso = px.data.gapminder()[["country", "iso_alpha"]]
    return uhi, flood, plastic, iso

def api_explain(module, payload):
    try:
        r = requests.post(f"{API_BASE}/explain/{module}", json=payload, timeout=20)
        return r.json()
    except Exception:
        return None

def api_post(endpoint, payload):
    try:
        r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=10)
        return r.json()
    except Exception:
        return {}

def get_lat_lon_cols(df):
    if "Latitude" in df.columns:
        return "Latitude", "Longitude"
    if "latitude" in df.columns:
        return "latitude", "longitude"
    raise ValueError("Latitude/Longitude columns not found")

def assign_region(df):
    lat_col, _ = get_lat_lon_cols(df)
    return pd.cut(
        df[lat_col],
        bins=[-90, -30, 0, 30, 60, 90],
        labels=["South", "Tropics", "Equatorial", "Subtropical", "North"]
    )

# ==============================
# EXPLANATIONS (RULE-BASED)
# ==============================

def explain_uhi(v):
    if v > 0.7:
        return "High UHI due to dense built-up areas and lack of vegetation."
    if v > 0.4:
        return "Moderate UHI influenced by mixed land cover."
    return "Low UHI supported by vegetation and reflective surfaces."

def explain_flood(v):
    if v > 0.7:
        return "High flood risk due to poor drainage and surface runoff."
    if v > 0.4:
        return "Moderate flood risk influenced by rainfall and terrain."
    return "Low flood risk with adequate drainage."

def explain_plastic(v):
    if v > 0.7:
        return "High plastic waste accumulation with low recycling."
    if v > 0.4:
        return "Moderate plastic waste generation."
    return "Plastic waste is relatively controlled."

def ensure_lat_lon(df, lat_col="Latitude", lon_col="Longitude"):
    """
    Ensure dataset has lat/lon.
    If missing, generate simulated coordinates for visualization.
    """
    if lat_col in df.columns and lon_col in df.columns:
        return df

    df = df.copy()
    df["Latitude"] = (
        pd.factorize(df.index)[0] * 0.5
    ) % 90 - 45

    df["Longitude"] = (
        pd.factorize(df.index)[0] * 0.7
    ) % 180 - 90

    return df

def get_environment_suggestions(country, uhi, flood, plastic):
    if groq_client is None:
        return "Groq AI not enabled. Set GROQ_API_KEY in .env."

    prompt = f"""
You are an environmental sustainability expert advising urban planners.

Country: {country}
Urban Heat Island intensity: {uhi}
Flood risk probability: {flood}
Plastic waste risk: {plastic}

Suggest 3–5 realistic, actionable measures to improve environmental conditions.
Do NOT mention AI, ML, or models.
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Groq error: {e}"

# ==============================
# LOAD DATA
# ==============================

uhi_df, flood_df, plastic_df, iso_map = load_data()
uhi_df["Region"] = assign_region(uhi_df)

if "Country" in plastic_df.columns:
    plastic_df = plastic_df.merge(
        iso_map,
        left_on="Country",
        right_on="country",
        how="left"
    )

# ==============================
# SESSION STATE
# ==============================

if "pinned_points" not in st.session_state:
    st.session_state.pinned_points = []

# ==============================
# SIDEBAR
# ==============================

st.sidebar.header("🌆 Controls")

regions = ["All Regions"] + sorted(uhi_df["Region"].dropna().unique().tolist())
selected_region = st.sidebar.selectbox("Select Region", regions)

uhi_view = uhi_df if selected_region == "All Regions" else uhi_df[uhi_df["Region"] == selected_region]

# ==============================
# HEADER
# ==============================

st.title("🌍 EcoVision – Environmental Risk Dashboard")
st.markdown(
    "Explainable analysis of **Urban Heat Island**, **Flood Risk**, "
    "and **Plastic Waste**, with AI-based sustainability recommendations."
)
st.markdown("---")

tab1, tab2, tab3 = st.tabs(
    ["🗺️ Risk Maps", "🧮 Predict", "📍 Pin & Predict"]
)

# ======================================================
# TAB 1 — MAPS (UHI FIXED)
# ======================================================

with tab1:
    st.subheader("🌡️ Urban Heat Island Map")

    lat_col, lon_col = get_lat_lon_cols(uhi_view)

    # ✅ FIXED: scatter_map (NOT density_map)
    fig = px.scatter_map(
        uhi_view,
        lat=lat_col,
        lon=lon_col,
        color="UHI",
        color_continuous_scale="hot",
        zoom=2 if selected_region == "All Regions" else 4,
        height=520
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("🌊 Flood Risk Distribution")
    st.plotly_chart(
        px.histogram(flood_df, x="FloodProbability", nbins=30),
        width="stretch"
    )

    st.subheader("♻️ Plastic Waste — Country Map (OSM)")

    import json

    geo_path = os.path.join(PROJECT_ROOT, "data", "world_countries.geo.json")

    if not os.path.exists(geo_path):
        st.warning("GeoJSON file missing: world_countries.geo.json")
    else:
        plastic_map = plastic_df.copy()

        if "PlasticRisk" not in plastic_map.columns and "Total_Plastic_Waste_MT" in plastic_map.columns:
            plastic_map["PlasticRisk"] = (
                plastic_map["Total_Plastic_Waste_MT"]
                / plastic_map["Total_Plastic_Waste_MT"].max()
            )

        world_geo = json.load(open(geo_path))
        value_map = plastic_map.set_index("Country")["PlasticRisk"].to_dict()

        for feature in world_geo["features"]:
            country = feature["properties"]["name"]
            feature["properties"]["PlasticRisk"] = (
                round(value_map.get(country, 0), 3)
                if country in value_map
                else "No data"
            )

        m = folium.Map(
            location=[20, 0],
            zoom_start=2,
            min_zoom=2,
            tiles=None,
            control_scale=True
        )

        folium.TileLayer(
            "OpenStreetMap",
            no_wrap=True
        ).add_to(m)

        # Choropleth (colors)
        choropleth = folium.Choropleth(
            geo_data=world_geo,
            data=plastic_map,
            columns=["Country", "PlasticRisk"],
            key_on="feature.properties.name",
            fill_color="YlGn",
            fill_opacity=0.7,
            line_opacity=0.2,
            legend_name="Plastic Waste Risk"
        ).add_to(m)

        # Create lookup dictionary
        risk_lookup = dict(zip(plastic_map["Country"], plastic_map["PlasticRisk"]))

        def style_function(feature):
            country = feature["properties"]["name"]
            value = risk_lookup.get(country, None)

            return {
                "fillOpacity": 0,  # invisible overlay
                "weight": 0
            }

        def tooltip_function(feature):
            country = feature["properties"]["name"]
            value = risk_lookup.get(country, "No data")

            return folium.GeoJsonTooltip(
                fields=[],
                aliases=[],
                labels=False
            )

        folium.GeoJson(
            world_geo,
            style_function=lambda x: {"fillOpacity": 0, "weight": 0},
            tooltip=folium.GeoJsonTooltip(
                fields=["name", "PlasticRisk"],
                aliases=["Country:", "Plastic Risk:"],
                sticky=True
            )
        ).add_to(m)

        # Lock bounds AFTER layers
        m.options["maxBounds"] = [[-60, -180], [85, 180]]

        st_folium(m, height=520, width="100%", returned_objects=[])


with tab2:
    st.subheader("🧮 Environmental Prediction")

    # ================= UHI INPUT =================
    st.markdown("### 🌡️ UHI Inputs")

    LST = st.slider("LST", 0.0, 1.0, 0.6)
    NDVI = st.slider("NDVI", 0.0, 1.0, 0.3)
    NDBI = st.slider("NDBI", 0.0, 1.0, 0.7)
    Albedo = st.slider("Albedo", 0.0, 1.0, 0.4)
    LULC = st.selectbox("LULC", [0, 1, 2, 3])

    # ================= FLOOD INPUT =================
    st.markdown("### 🌊 Flood Inputs")

    PopulationScore = st.slider("PopulationScore", 0.0, 1.0, 0.5)
    WetlandLoss = st.slider("WetlandLoss", 0.0, 1.0, 0.5)
    InadequatePlanning = st.slider("InadequatePlanning", 0.0, 1.0, 0.5)
    PoliticalFactors = st.slider("PoliticalFactors", 0.0, 1.0, 0.5)

    # ================= PLASTIC INPUT =================
    st.markdown("### ♻️ Plastic Inputs")

    Country = st.slider("Country index", 0, 200, 50)
    Main_Sources = st.slider("Main Sources", 0, 5, 2)
    Coastal_Waste_Risk = st.slider("Coastal Waste Risk", 0, 5, 1)
    Total_Plastic_Waste_MT = st.slider("Total Plastic Waste", 0.0, 1.0, 0.5)
    Per_Capita_Waste_KG = st.slider("Per Capita Waste", 0.0, 1.0, 0.5)
    Recycling_Rate = st.slider("Recycling Rate", 0.0, 1.0, 0.4)

    if st.button("Predict Environment"):

        # ================= CALL API =================
        uhi_payload = {
            "LST": LST,
            "NDVI": NDVI,
            "NDBI": NDBI,
            "Albedo": Albedo,
            "LULC": LULC
        }

        flood_payload = {
            "PopulationScore": PopulationScore,
            "WetlandLoss": WetlandLoss,
            "InadequatePlanning": InadequatePlanning,
            "PoliticalFactors": PoliticalFactors
        }

        plastic_payload = {
            "Country": Country,
            "Main_Sources": Main_Sources,
            "Coastal_Waste_Risk": Coastal_Waste_Risk,
            "Total_Plastic_Waste_MT": Total_Plastic_Waste_MT,
            "Per_Capita_Waste_KG": Per_Capita_Waste_KG,
            "Recycling_Rate": Recycling_Rate
        }

        uhi_res = api_post("/predict/uhi", uhi_payload)
        flood_res = api_post("/predict/flood", flood_payload)
        plastic_res = api_post("/predict/plastic", plastic_payload)

        uhi_val = uhi_res.get("UHI", 0)
        flood_val = flood_res.get("FloodProbability", 0)
        plastic_val = plastic_res.get("PlasticWasteRisk", 0)

        # ================= DISPLAY =================
        c1, c2, c3 = st.columns(3)
        c1.metric("🌡️ UHI", round(uhi_val, 3))
        c2.metric("🌊 Flood", round(flood_val, 3))
        c3.metric("♻️ Plastic", round(plastic_val, 3))

        st.markdown("### 🧠 Explanation")
        st.info(explain_uhi(uhi_val))
        st.info(explain_flood(flood_val))
        st.info(explain_plastic(plastic_val))

        # ================= SHAP =================
        st.markdown("### 🧠 SHAP Explanation")

        shap_uhi = api_explain("uhi", uhi_payload)
        shap_flood = api_explain("flood", flood_payload)
        shap_plastic = api_explain("plastic", plastic_payload)

        for name, data in {
            "UHI": shap_uhi,
            "Flood": shap_flood,
            "Plastic": shap_plastic
        }.items():

            if data and "values" in data:
                st.markdown(f"**{name}**")

                df = pd.DataFrame({
                    "Feature": list(data["values"].keys()),
                    "Impact": list(data["values"].values())
                })

                st.bar_chart(df.set_index("Feature"))
                st.write(data.get("text", ""))

        # ================= GROQ =================
        st.markdown("### 🛠️ Improvement Suggestions (Groq AI)")

        suggestions = get_environment_suggestions(
            "Selected Area",
            uhi_val,
            flood_val,
            plastic_val
        )

        st.markdown(suggestions)

with tab3:
    st.subheader("📍 Pin Locations & Predict")

    geolocator = Nominatim(user_agent="ecovision_app")

    m = folium.Map(location=[0, 0], zoom_start=2)

    for p in st.session_state.pinned_points:
        folium.Marker([p["lat"], p["lon"]], popup=p["country"]).add_to(m)

    map_data = st_folium(m, height=520, width="100%")

    if map_data and map_data.get("last_clicked"):
        lat = map_data["last_clicked"]["lat"]
        lon = map_data["last_clicked"]["lng"]

        try:
            country = geolocator.reverse((lat, lon)).raw["address"].get("country", "Unknown")
        except:
            country = "Unknown"

        st.session_state.pinned_points.append({"lat":lat,"lon":lon,"country":country})
        st.rerun()

    if st.session_state.pinned_points:
        idx = st.selectbox("Select pinned location", range(len(st.session_state.pinned_points)))
        p = st.session_state.pinned_points[idx]

        if st.button("Predict for this location"):
            uhi_v = api_post("/predict/uhi", {"LST":0.5,"NDVI":0.5,"NDBI":0.5,"Albedo":0.4,"LULC":2}).get("UHI",0)
            flood_v = api_post("/predict/flood", {"PopulationScore":0.5,"WetlandLoss":0.5,"InadequatePlanning":0.5,"PoliticalFactors":0.5}).get("FloodProbability",0)
            plastic_v = api_post("/predict/plastic", {"Country":10,"Main_Sources":2,"Coastal_Waste_Risk":1,"Total_Plastic_Waste_MT":0.5,"Per_Capita_Waste_KG":0.5,"Recycling_Rate":0.5}).get("PlasticWasteRisk",0)
            # ---- SHAP CALLS ----
            p["shap"] = {}

            uhi_payload = {
                "LST": abs(p["lat"] % 1),
                "NDVI": 1 - abs(p["lat"] % 1),
                "NDBI": abs(p["lat"] % 1),
                "Albedo": 0.4,
                "LULC": 2
            }

            flood_payload = {
                "PopulationScore": min(abs(p["lat"]) / 90, 1),
                "WetlandLoss": min(abs(p["lon"]) / 180, 1),
                "InadequatePlanning": 0.6,
                "PoliticalFactors": 0.5
            }

            plastic_payload = {
                "Country": 10,
                "Main_Sources": 2,
                "Coastal_Waste_Risk": 1,
                "Total_Plastic_Waste_MT": 0.6,
                "Per_Capita_Waste_KG": 0.5,
                "Recycling_Rate": 0.4
            }

            p["shap"]["uhi"] = api_explain("uhi", uhi_payload)
            p["shap"]["flood"] = api_explain("flood", flood_payload)
            p["shap"]["plastic"] = api_explain("plastic", plastic_payload)
            p["prediction"] = {"uhi":uhi_v,"flood":flood_v,"plastic":plastic_v}
            p["suggestions"] = get_environment_suggestions(p["country"], uhi_v, flood_v, plastic_v)

        if p.get("prediction"):
            st.metric("UHI", p["prediction"]["uhi"])
            st.metric("Flood", p["prediction"]["flood"])
            st.metric("Plastic", p["prediction"]["plastic"])

        if p.get("suggestions"):
            st.markdown("### 🛠️ Suggestions")
            st.markdown(p["suggestions"])
        if p.get("shap"):
            st.markdown("### 🧠 SHAP Explanation")

            for module, data in p["shap"].items():
                if data:
                    st.markdown(f"**{module.upper()}**")

                    df = pd.DataFrame({
                        "Feature": list(data["values"].keys()),
                        "Impact": list(data["values"].values())
                    })

                    st.bar_chart(df.set_index("Feature"))
                    st.write(data.get("text", ""))


st.markdown("---")
st.markdown(
    "<center><b>EcoVision</b> – Explainable Environmental Intelligence</center>",
    unsafe_allow_html=True
)
