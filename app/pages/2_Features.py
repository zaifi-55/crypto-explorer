import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.features import build_features

st.set_page_config(
    page_title="Feature Engineering"
)

st.title("Feature Engineering")

# --- Load data ---
df = pd.read_parquet("data/raw/ohlcv.parquet")

# --- Build features ---
df = build_features(df)

# --- Figure out which column is the date/timestamp ---
date_col = "timestamp" if "timestamp" in df.columns else "date"

# --- Feature selection options ---
features_options = [
    col for col in df.columns 
    if col not in [date_col, "close", "open", "high", "low", "volume"]
]
selected_features = st.multiselect(
    "Select features to include",
    features_options,
    default=features_options
)

# --- Subset dataframe ---
df_selected = df[[date_col, "close"] + selected_features]

# --- Show preview ---
st.write("### Preview of Features")
st.dataframe(df_selected.tail())

# --- Correlation heatmap ---
if st.checkbox("Show correlation heatmap"):
    corr = df_selected.drop(columns=[date_col]).corr()
    fig, ax = plt.subplots(figsize=(10, 6))
    cax = ax.matshow(corr, cmap="coolwarm")
    fig.colorbar(cax)
    st.pyplot(fig)
