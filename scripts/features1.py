import pandas as pd
import numpy as np
import os

def create_features_for_web_interface():
    """
    Create a features file that the web interface can use.
    """
    # Load the raw data
    input_path = "data/processed/bitcoin.csv"
    output_path = "data/processed/bitcoin_features.csv"
    
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    
    print(f"Original shape: {df.shape}")
    print(f"Original columns: {list(df.columns)}")
    
    # Convert date and sort
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date').reset_index(drop=True)
    
    # Create lagged features (previous prices)
    for lag in [1, 2, 3, 5, 7, 14]:
        df[f'price_lag_{lag}'] = df['price'].shift(lag)
    
    # Create moving averages
    for window in [3, 7, 14, 30]:
        df[f'ma_{window}'] = df['price'].rolling(window=window).mean()
    
    # Create returns and volatility
    df['price_change'] = df['price'].pct_change()
    df['price_change_lag_1'] = df['price_change'].shift(1)
    df['volatility_7d'] = df['price_change'].rolling(window=7).std()
    df['volatility_30d'] = df['price_change'].rolling(window=30).std()
    
    # Create price ratios
    df['price_to_ma_7'] = df['price'] / df['ma_7']
    df['price_to_ma_30'] = df['price'] / df['ma_30']
    
    # Create momentum features
    df['momentum_5d'] = df['price'] / df['price_lag_5'] - 1
    df['momentum_14d'] = df['price'] / df['price_lag_14'] - 1
    
    # Create RSI (Relative Strength Index)
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Create target variables (different options)
    df['target_next_price'] = df['price'].shift(-1)  # Next day price
    df['target_return'] = df['price'].shift(-1) / df['price'] - 1  # Next day return
    df['target_direction'] = (df['price'].shift(-1) > df['price']).astype(int)  # Up/Down
    
    # Remove rows with NaN values
    df_clean = df.dropna()
    
    print(f"Final shape after feature engineering: {df_clean.shape}")
    print(f"Final columns: {list(df_clean.columns)}")
    
    # Save the processed data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    
    print(f"\nFeatures file saved to: {output_path}")
    print(f"Removed {len(df) - len(df_clean)} rows due to NaN values")
    
    # Print feature summary
    feature_cols = [col for col in df_clean.columns if col not in ['date', 'target_next_price', 'target_return', 'target_direction']]
    print(f"\nCreated {len(feature_cols)} features:")
    for i, col in enumerate(feature_cols):
        if i < 10:  # Show first 10
            print(f"  - {col}")
        elif i == 10:
            print(f"  ... and {len(feature_cols) - 10} more")
            break
    
    print(f"\nTarget options:")
    print(f"  - target_next_price: Next day's price")
    print(f"  - target_return: Next day's return (percentage change)")
    print(f"  - target_direction: Up (1) or Down (0) direction")
    
    return output_path

if __name__ == "__main__":
    create_features_for_web_interface()