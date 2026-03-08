import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
from src.api import fetch_ohlc

os.makedirs("data/raw", exist_ok=True)


df = fetch_ohlc("bitcoin", 90, vs_currency="usd")

output_path = "data/raw/ohlcv.parquet"
df.to_parquet(output_path, index=False)

print(f"✅ Saved {df.shape[0]} rows to {output_path}")
