import os
import argparse
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.modeling import train_models_pipeline
from src.config import COIN_MAP, PROCESSED_DIR, MODELS_DIR


def resolve_coin_name(arg_coin: str) -> str:
    """
    Maps input coin argument (ticker or name) to the processed filename base.
    """
    arg_coin = arg_coin.lower()
    if arg_coin in COIN_MAP:
        return COIN_MAP[arg_coin].lower()
    return arg_coin


def create_basic_features(df):
    """
    Create basic features if only price data is available.
    """
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').reset_index(drop=True)
    
    # Create lagged features
    for lag in [1, 2, 3, 5]:
        df[f'price_lag_{lag}'] = df['price'].shift(lag)
    
    # Create moving averages
    for window in [3, 7, 14]:
        df[f'ma_{window}'] = df['price'].rolling(window=window).mean()
    
    # Create price changes
    df['price_change'] = df['price'].pct_change()
    df['price_change_lag_1'] = df['price_change'].shift(1)
    
    # Create target: predict next day's return
    df['target'] = df['price'].shift(-1) / df['price'] - 1
    
    # Remove rows with NaN
    df = df.dropna()
    
    # Set date as index if available
    if 'date' in df.columns:
        df.set_index('date', inplace=True)
    
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coin", required=True, help="Coin ticker or name (e.g. BTCUSDT or bitcoin)")
    parser.add_argument("--target", default="target", help="Target column to predict")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of CV splits")
    parser.add_argument("--metric", default="rmse", choices=["rmse", "mae"], help="Primary metric for model selection")
    parser.add_argument("--auto_features", action="store_true", help="Automatically create features from price data")
    args = parser.parse_args()

    # Normalize coin name
    coin_name = resolve_coin_name(args.coin)
    base = os.path.join(PROCESSED_DIR, coin_name)

    # Prefer parquet, else csv
    if os.path.exists(base + ".parquet"):
        path = base + ".parquet"
    elif os.path.exists(base + ".csv"):
        path = base + ".csv"
    else:
        available = [f for f in os.listdir(PROCESSED_DIR) if f.endswith((".csv", ".parquet"))]
        raise FileNotFoundError(
            f"No processed file found for {coin_name}. "
            f"Looked for {base}.csv or {base}.parquet. "
            f"Available files: {available}"
        )

    print(f"Loading data from: {path}")
    
    # Load data
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Check if we need to create features
    if args.target not in df.columns:
        if args.auto_features and 'price' in df.columns:
            print(f"Target '{args.target}' not found. Creating features automatically...")
            df = create_basic_features(df)
            print(f"New shape after feature engineering: {df.shape}")
            print(f"New columns: {list(df.columns)}")
        else:
            print(f"Error: Target column '{args.target}' not found.")
            print(f"Available columns: {list(df.columns)}")
            print("Tip: Use --auto_features flag to automatically create features from price data")
            print("Or specify --target price to use price directly")
            return

    # Check if we have enough data after feature engineering
    if len(df) < 50:
        print(f"Warning: Only {len(df)} samples remaining after feature engineering.")
        print("Consider using more historical data.")
        return

    # Train and evaluate models
    results = train_models_pipeline(
        df=df, 
        target=args.target, 
        n_splits=args.n_splits,
        primary_metric=args.metric,
        coin_name=coin_name
    )

    # Save results
    os.makedirs(MODELS_DIR, exist_ok=True)
    out_path = os.path.join(MODELS_DIR, f"{coin_name}_results.csv")
    results.to_csv(out_path, index=False)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()