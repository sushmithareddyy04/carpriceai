import sys
import types
import math
import os
import json
import pickle

# --- Compatibility Layer for Windows Application Control (SAC/WDAC) ---
HAS_NUMPY = True
try:
    import numpy as np
except Exception:
    HAS_NUMPY = False

    class MockNumpy(types.ModuleType):
        __version__ = "1.26.4"
        ndarray = type("ndarray", (), {"shape": (1,), "__len__": lambda self: 1})
        nan = float("nan")
        inf = float("inf")
        pi = math.pi
        float64 = float
        float32 = float
        float_ = float
        int64 = int
        int32 = int
        int_ = int
        bool_ = bool
        dtype = lambda x: str(x)

        @staticmethod
        def isscalar(val):
            return isinstance(val, (int, float, str, bool, bytes))

        @staticmethod
        def isnan(val):
            return math.isnan(val) if isinstance(val, (int, float)) else False

        @staticmethod
        def isinf(val):
            return math.isinf(val) if isinstance(val, (int, float)) else False

        @staticmethod
        def array(data, *args, **kwargs):
            if hasattr(data, "__iter__") and not isinstance(data, (str, bytes, dict)):
                return list(data)
            return [data]

        @staticmethod
        def asarray(data, *args, **kwargs):
            return MockNumpy.array(data)

    mock_np = MockNumpy("numpy")
    sys.modules["numpy"] = mock_np
    np = mock_np

HAS_PANDAS = True
try:
    import pandas as pd
except Exception:
    HAS_PANDAS = False

    class MockPandas(types.ModuleType):
        __version__ = "2.2.2"
        DataFrame = type("DataFrame", (), {})
        Series = type("Series", (), {})
        Index = type("Index", (), {})
        NA = None
        NaT = None
        class api:
            class extensions:
                ExtensionArray = type("ExtensionArray", (), {})

    mock_pd = MockPandas("pandas")
    sys.modules["pandas"] = mock_pd
    pd = mock_pd

import streamlit as st
from src.inference_engine import ValuationEngine

# --- Page Configuration ---
st.set_page_config(
    page_title="CarPriceAI — Used Car Price Prediction",
    page_icon="🚗",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# --- Clean, Modern Styling ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, sans-serif;
    }
    .main {
        background-color: #f8fafc;
    }
    .main-header {
        text-align: center;
        padding: 20px 0 10px 0;
    }
    .main-header h1 {
        font-size: 2.4rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 6px;
    }
    .main-header p {
        font-size: 1.05rem;
        color: #64748b;
        margin: 0;
    }
    .prediction-box {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: white;
        padding: 28px;
        border-radius: 16px;
        text-align: center;
        box-shadow: 0 10px 25px -5px rgba(15, 23, 42, 0.25);
        margin: 25px 0;
    }
    .prediction-price {
        font-size: 3rem;
        font-weight: 800;
        color: #38bdf8;
        margin: 10px 0;
    }
    .segment-pill {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }
    .seg-economy {
        background-color: #dcfce7;
        color: #15803d;
    }
    .seg-midrange {
        background-color: #e0f2fe;
        color: #0369a1;
    }
    .seg-luxury {
        background-color: #fef3c7;
        color: #b45309;
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        background: white;
        padding: 16px;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        margin-top: 15px;
        text-align: center;
    }
    .metric-item h4 {
        margin: 0;
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
    }
    .metric-item p {
        margin: 4px 0 0 0;
        font-size: 1.15rem;
        font-weight: 700;
        color: #0f172a;
    }
    </style>
""", unsafe_allow_html=True)

# Load Model Engine
@st.cache_resource
def get_engine():
    return ValuationEngine(artifacts_dir="models")

engine = get_engine()

# --- Header ---
st.markdown("""
    <div class="main-header">
        <h1>🚗 CarPriceAI</h1>
        <p>Advanced Used Car Price Prediction System</p>
    </div>
""", unsafe_allow_html=True)

# Quick Presets
preset = None
col_p1, col_p2, col_p3, col_p4 = st.columns(4)
if col_p1.button("🚗 Creta (2021)", use_container_width=True):
    preset = ("Hyundai Creta SX Diesel", 2021, 45000, "Diesel", "Automatic", "Individual", "First Owner", 1493, 113.4, 18.5)
if col_p2.button("🚙 Swift (2018)", use_container_width=True):
    preset = ("Maruti Swift Dzire VDI", 2018, 62000, "Diesel", "Manual", "Individual", "Second Owner", 1248, 74.0, 26.5)
if col_p3.button("🏎️ BMW 5 Series (2019)", use_container_width=True):
    preset = ("BMW 5 Series 520d", 2019, 38000, "Diesel", "Automatic", "Dealer", "First Owner", 1995, 187.7, 17.4)
if col_p4.button("🛻 Fortuner (2022)", use_container_width=True):
    preset = ("Toyota Fortuner 2.8 4x4", 2022, 35000, "Diesel", "Automatic", "First Owner", "First Owner", 2755, 201.1, 14.2)

# Default Values
d_name = preset[0] if preset else "Hyundai Creta SX Diesel"
d_year = preset[1] if preset else 2021
d_km = preset[2] if preset else 45000
d_fuel = preset[3] if preset else "Diesel"
d_trans = preset[4] if preset else "Automatic"
d_seller = preset[5] if preset else "Individual"
d_owner = preset[6] if preset else "First Owner"
d_engine = preset[7] if preset else 1493
d_power = preset[8] if preset else 113.4
d_mileage = preset[9] if preset else 18.5

# --- Main Prediction Form ---
with st.form("price_predictor_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        car_name = st.text_input("Car Make & Model", d_name)
        year = st.slider("Manufacturing Year", 2008, 2025, d_year)
        km_driven = st.number_input("Kilometers Driven (KM)", 1000, 300000, d_km, step=5000)
        fuel = st.selectbox("Fuel Type", ["Diesel", "Petrol", "CNG", "LPG", "Electric"], index=["Diesel", "Petrol", "CNG", "LPG", "Electric"].index(d_fuel))
        transmission = st.selectbox("Transmission", ["Manual", "Automatic"], index=["Manual", "Automatic"].index(d_trans))
        
    with col2:
        seller_type = st.selectbox("Seller Type", ["Individual", "Dealer", "Trustmark Dealer"], index=["Individual", "Dealer", "Trustmark Dealer"].index(d_seller) if d_seller in ["Individual", "Dealer", "Trustmark Dealer"] else 0)
        owner = st.selectbox("Ownership History", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"], index=["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner"].index(d_owner))
        engine_cc = st.slider("Engine (CC)", 750, 4000, d_engine, step=50)
        max_power = st.slider("Max Power (BHP)", 40.0, 350.0, float(d_power), step=2.5)
        mileage = st.slider("Mileage (kmpl)", 8.0, 35.0, float(d_mileage), step=0.5)

    submitted = st.form_submit_button("⚡ Predict Estimated Resale Price", use_container_width=True)

# Run Inference (on submit or by default)
if submitted or preset:
    val = engine.predict_price({
        "name": car_name,
        "year": year,
        "km_driven": km_driven,
        "fuel": fuel,
        "transmission": transmission,
        "seller_type": seller_type,
        "owner": owner,
        "engine": engine_cc,
        "max_power": max_power,
        "mileage": mileage,
        "seats": 5
    })

    price = val["estimated_price"]
    low = val["lower_bound"]
    high = val["upper_bound"]
    seg = val["market_segment"]

    seg_class = "seg-midrange"
    if "Luxury" in seg:
        seg_class = "seg-luxury"
    elif "Economy" in seg:
        seg_class = "seg-economy"

    # Prominent Price Display
    st.markdown(f"""
        <div class="prediction-box">
            <span class="segment-pill {seg_class}">{seg}</span>
            <div style="font-size: 1rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;">Estimated Market Value</div>
            <div class="prediction-price">₹ {price:,.0f}</div>
            <div style="color: #cbd5e1; font-size: 1.05rem;">
                Estimated Range: <b>₹ {low:,.0f}</b> — <b>₹ {high:,.0f}</b>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Key Metrics Bar
    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-item">
                <h4>Original Est. MRP</h4>
                <p>₹ {val['estimated_new_mrp']:,.0f}</p>
            </div>
            <div class="metric-item">
                <h4>Total Depreciation</h4>
                <p style="color: #ef4444;">-{val['depreciation_percent']}%</p>
            </div>
            <div class="metric-item">
                <h4>Vehicle Age</h4>
                <p>{val['vehicle_age']} Years</p>
            </div>
            <div class="metric-item">
                <h4>Odometer</h4>
                <p>{km_driven:,.0f} KM</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

# Clean, subtle footer
st.markdown("""
    <div style="text-align: center; margin-top: 40px; color: #94a3b8; font-size: 0.85rem;">
        CarPriceAI • Powered by Machine Learning Regression Ensemble
    </div>
""", unsafe_allow_html=True)
