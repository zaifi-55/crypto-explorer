import os
import sys
import pandas as pd
import streamlit as st

# ensure project root on path (so we can import src.* when run from this page)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import PROCESSED_DIR, MODELS_DIR
from src.modeling import (
    make_tscv, cross_validate_models, split_date_ranges,
    build_metadata, save_model_bundle
)

st.set_page_config(page_title="Train Models", layout="wide")
st.title("Model Training")

def create_features_from_price(df):
    """Create features from basic price data"""
    df = df.copy()
    
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    
    if 'price' not in df.columns:
        st.error("Price column not found!")
        return df
    
    # Create lagged features
    for lag in [1, 2, 3, 5]:
        df[f'price_lag_{lag}'] = df['price'].shift(lag)
    
    # Create moving averages
    for window in [3, 7, 14]:
        df[f'ma_{window}'] = df['price'].rolling(window=window).mean()
    
    # Create price changes
    df['price_change'] = df['price'].pct_change()
    df['price_change_lag_1'] = df['price_change'].shift(1)
    
    # Create momentum features
    df['momentum_5'] = df['price'] / df['price'].shift(5) - 1
    df['volatility_7'] = df['price'].rolling(7).std()
    
    # Create target: next day's return
    df['target'] = df['price'].shift(-1) / df['price'] - 1
    
    # Remove NaN rows
    df = df.dropna()
    
    # Set date as index
    if 'date' in df.columns:
        df = df.set_index('date')
    
    return df

# ---- sidebar controls
st.sidebar.header("Dataset")

# list processed files (*.csv|*.parquet)
processed_files = []
if os.path.exists(PROCESSED_DIR):
    for fn in os.listdir(PROCESSED_DIR):
        if fn.endswith(".csv") or fn.endswith(".parquet"):
            processed_files.append(fn)
processed_files = sorted(processed_files)

if not processed_files:
    st.warning("No processed files found in /data/processed. Create one from the previous steps.")
    st.stop()

file_choice = st.sidebar.selectbox("Choose processed file", processed_files)
coin_name = os.path.splitext(file_choice)[0]

path = os.path.join(PROCESSED_DIR, file_choice)
try:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
except Exception as e:
    st.error(f"Error loading file: {e}")
    st.stop()

# assume 'date' exists; if not, let the index as-is
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")

st.write(f"**Loaded:** `{file_choice}` rows={len(df)}")
st.write(f"**Columns:** {list(df.columns)}")

# Check if we need to create features
original_columns = list(df.columns)
needs_features = len(df.columns) <= 2 and 'price' in df.columns

if needs_features:
    st.info("🔧 Basic price data detected. Auto-generating features for machine learning...")
    
    with st.expander("Feature Engineering Options", expanded=True):
        create_features = st.checkbox("Create features automatically", value=True, 
                                    help="Generate lagged prices, moving averages, and returns")
        
        if create_features:
            with st.spinner("Creating features..."):
                df = create_features_from_price(df)
            
            st.success(f"✅ Features created! New shape: {df.shape}")
            st.write("**New columns:**", list(df.columns))
            
            # Show feature preview
            with st.expander("Preview Generated Features"):
                st.dataframe(df.head())

# Target/Features selection
all_cols = list(df.columns)
default_target = "target" if "target" in all_cols else all_cols[-1]

st.sidebar.subheader("Model Configuration")
target_col = st.sidebar.selectbox("Target column", all_cols, 
                                index=all_cols.index(default_target))

# Available features (exclude target and date-like columns)
available_features = [c for c in all_cols if c != target_col and c != 'date']

if not available_features:
    st.error("❌ No features available for training!")
    st.write("**Available columns:**", all_cols)
    st.write("**Selected target:**", target_col)
    st.write("💡 **Tip:** If you only have price data, enable 'Create features automatically' above.")
    st.stop()

# Feature selection
feature_default = available_features  # Select all by default
feature_cols = st.sidebar.multiselect(
    "Features",
    options=available_features,
    default=feature_default,
    help=f"Select from {len(available_features)} available features"
)

# Cross-validation parameters
n_splits = st.sidebar.slider("TimeSeriesSplit folds", min_value=3, max_value=10, value=5)
gap = st.sidebar.number_input("Gap between train/test (optional)", min_value=0, value=0, step=1)

# Show data info
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Samples", len(df))
with col2:
    st.metric("Features Selected", len(feature_cols))
with col3:
    st.metric("CV Folds", n_splits)

# Run button
run_cv = st.button("Run Cross-Validation", type="primary", disabled=len(feature_cols) == 0)

if run_cv:
    if not feature_cols:
        st.error("Please select at least one feature.")
        st.stop()

    with st.spinner("Training models..."):
        X = df[feature_cols].copy()
        y = df[target_col].copy()

        # Show data summary
        st.write("### Training Data Summary")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Features shape:**", X.shape)
            st.write("**Target shape:**", y.shape)
        with col2:
            st.write("**Target statistics:**")
            st.write(y.describe())

        tscv = make_tscv(n_splits=n_splits, gap=gap)
        results, raw = cross_validate_models(X, y, tscv, primary_metric="rmse")

        # comparison table
        table = pd.DataFrame([{
            "Model": r.model_name,
            "RMSE (cv)": f"{r.mean_rmse:.6f}",
            "MAE (cv)": f"{r.mean_mae:.6f}",
            "R² (cv)": f"{r.mean_r2:.4f}",
            "Best Params": str(r.best_params)[:100] + "..." if len(str(r.best_params)) > 100 else str(r.best_params)
        } for r in results])

        st.subheader("🏆 Model Comparison Results")
        st.write("*Lower RMSE is better*")
        st.dataframe(table, use_container_width=True)

        # store best in session
        st.session_state["best_result"] = results[0]
        st.session_state["tscv_ranges"] = split_date_ranges(df.index, tscv)
        st.session_state["coin_name"] = coin_name
        st.session_state["feature_cols"] = feature_cols
        st.session_state["target_col"] = target_col

        # Show best model details
        best = results[0]
        st.success(f"🥇 Best Model: **{best.model_name}** (RMSE: {best.mean_rmse:.6f})")

# Save section
st.divider()
st.subheader("💾 Save Best Model")

if "best_result" not in st.session_state:
    st.info("👆 Run cross-validation first to train models")
else:
    col1, col2 = st.columns([1, 3])
    with col1:
        save_clicked = st.button("Save Best to /models", type="primary")
    with col2:
        out_dir = st.text_input("Output folder", value=MODELS_DIR)

    if save_clicked:
        try:
            best = st.session_state["best_result"]
            ranges = st.session_state["tscv_ranges"]
            coin = st.session_state["coin_name"]

            meta = build_metadata(
                coin=coin,
                model_name=best.model_name,
                feature_cols=st.session_state["feature_cols"],
                target_col=st.session_state["target_col"],
                cv_date_ranges=ranges,
                best_params=best.best_params,
                scores={"rmse": best.mean_rmse, "mae": best.mean_mae, "r2": best.mean_r2},
                notes="saved via Streamlit page 3_Train.py with auto-generated features",
            )

            stem = f"{coin}_{best.model_name}"
            os.makedirs(out_dir, exist_ok=True)
            fullpath = save_model_bundle(best.estimator, meta, stem, out_dir=out_dir)
            st.success(f"✅ Model saved successfully!")
            st.code(f"Location: {fullpath}")
            
            with st.expander("Model Metadata"):
                st.json(meta)
                
        except Exception as e:
            st.error(f"Error saving model: {e}")