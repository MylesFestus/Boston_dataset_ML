import os
import pickle

import streamlit as st
import numpy as np
import pandas as pd

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(page_title="Boston Housing Price Prediction", layout="wide")

# =====================================================
# FEATURE NAME MAPPING
# =====================================================
FEATURE_NAME_MAP = {
    "CRIM": "Per capita crime rate",
    "ZN": "Residential land zoned %",
    "INDUS": "Non-retail business acres %",
    "CHAS": "Charles River dummy variable",
    "NOX": "Nitric oxides concentration",
    "RM": "Average number of rooms",
    "AGE": "Owner-occupied units built before 1940",
    "DIS": "Distance to employment centers",
    "RAD": "Accessibility to radial highways",
    "TAX": "Property tax rate",
    "PTRATIO": "Pupil-teacher ratio",
    "B": "Proportion of Black population index",
    "LSTAT": "Lower status population %",
}

# =====================================================
# LOAD MODEL
# =====================================================
def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaling.pkl"
DATA_PATH = "boston_housing.csv"
IMAGE_PATH = "house.jpg"

if not os.path.exists(MODEL_PATH):
    st.error("❌ model.pkl not found")
    st.stop()

model = load_pickle(MODEL_PATH)
scaler = load_pickle(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

if not os.path.exists(DATA_PATH):
    st.error("❌ boston_housing.csv not found")
    st.stop()

df = pd.read_csv(DATA_PATH)

# =====================================================
# FEATURE EXTRACTION
# =====================================================
def get_features(df):
    cols = list(df.columns)
    if "MEDV" in cols:
        return [c for c in cols if c != "MEDV"]
    return cols[:-1]

feature_cols = get_features(df)

# =====================================================
# HEADER (TITLE + IMAGE SIDE BY SIDE)
# =====================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.title("🏠 Boston Housing Price Prediction")
    st.markdown("Predict house prices using Machine Learning")

with col2:
    if os.path.exists(IMAGE_PATH):
        st.image(IMAGE_PATH, width=220)
    else:
        st.image(
            "https://images.unsplash.com/photo-1568605114967-8130f3a36994",
            width=220
        )

st.divider()

# =====================================================
# INPUT SECTION
# =====================================================
st.subheader("Enter House Features")

defaults = df[feature_cols].median().to_dict()

cols = st.columns(3)
user_input = {}

for i, f in enumerate(feature_cols):
    with cols[i % 3]:

        label = f"{FEATURE_NAME_MAP.get(f, f)} ({f})"

        user_input[f] = st.number_input(
            label,
            value=float(defaults.get(f, 0.0))
        )

X_user = pd.DataFrame([[user_input[c] for c in feature_cols]], columns=feature_cols)

# =====================================================
# PREDICTION FUNCTION
# =====================================================
def predict(model, X, scaler=None):
    X = X.values.astype(float)

    if scaler is not None and not hasattr(model, "steps"):
        X = scaler.transform(X)

    return float(model.predict(X)[0])

# =====================================================
# PREDICTION OUTPUT
# =====================================================
st.subheader("Prediction")

if st.button("Predict Price"):
    result = predict(model, X_user, scaler)
    st.success(f"🏠 Estimated House Price: ${result:.2f}k")
else:
    st.info("Fill values and click Predict")