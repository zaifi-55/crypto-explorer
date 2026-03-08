import streamlit as st
import pandas as pd
import glob
import plotly.express as px
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.baselines import evaluate_baselines

st.set_page_config(page_title="Train - Baselines", layout="wide")

def list_coins_from_processed(data_dir="data/processed"):
    files = glob.glob(os.path.join(data_dir, "*.csv")) + glob.glob(os.path.join(data_dir, "*.parquet"))
    coins = [os.path.basename(f).split('.')[0] for f in files]
    return coins, files

def load_features_for_coin(coin_name, data_dir="data/processed"):
    csv_path = os.path.join(data_dir, f"{coin_name}.csv")
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path, parse_dates=True, index_col=0)

    pq_path = os.path.join(data_dir, f"{coin_name}.parquet")
    if os.path.exists(pq_path):
        return pd.read_parquet(pq_path)

    candidates = glob.glob(os.path.join(data_dir, f"{coin_name}*"))
    if candidates:
        p = candidates[0]
        if p.endswith(".csv"):
            return pd.read_csv(p, parse_dates=True, index_col=0)
        else:
            return pd.read_parquet(p)

    raise FileNotFoundError(f"No processed file found for {coin_name}")

# ---------------- UI -----------------
coins, files = list_coins_from_processed()
coin = st.selectbox("Choose coin (processed)", coins)

target_col = st.text_input("Target column (price)", value="price")
test_pct = st.slider("Test set size (%)", min_value=10, max_value=30, value=20, step=5)
ma_windows_input = st.text_input("MA windows (comma separated)", value="3,7,14")
ma_windows = tuple(int(x.strip()) for x in ma_windows_input.split(",") if x.strip().isdigit())

if st.button("Run Baselines"):
    df = load_features_for_coin(coin)
    st.write(f"Loaded {len(df)} rows for {coin} - using target: '{target_col}'")

    # Use the dynamic target_col and other parameters from UI
    results_df, preds, train, val, test = evaluate_baselines(
        df,
        target_col=target_col,  # ← Fixed: use the UI input
        test_size=test_pct/100,  # ← Fixed: use the slider value
        ma_windows=list(ma_windows),  # ← Fixed: use the UI input
        val_size=0.1
    )

    st.subheader("Baseline metrics (on test split)")
    st.dataframe(results_df.set_index('model').round(6))

    st.subheader("Actual vs Predicted (test range)")
    for model_name, pred_series in preds.items():
        plot_df = pd.DataFrame({
            "actual": test,  # ← Fixed: test is already a Series of the target column
            model_name: pred_series
        }).dropna(how='all')

        if plot_df.empty:
            st.write(f"No overlap for {model_name} predictions on test set (probably NaNs)")
            continue

        fig = px.line(plot_df, x=plot_df.index, y=plot_df.columns,
                      labels={"value": "Price", "index": "Date"},
                      title=f"Actual vs {model_name} (test)")
        st.plotly_chart(fig, use_container_width=True)

    st.success("Done -- Baselines Evaluated.")