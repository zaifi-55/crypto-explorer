import pandas as pd
import os,glob

RAW_DIR="data/raw"
PROC_DIR="data/processed"
os.makedirs(PROC_DIR,exist_ok=True)

for f in glob.glob(os.path.join(RAW_DIR,"*.csv")):
    coin=os.path.basename(f).split(".")[0]
    df=pd.read_csv(f,parse_dates=True,index_col=0)

    if "date" in df.columns:
        df["date"]=pd.to_datetime(df["date"])
        df.set_index("date",inplace=True)


    out_path=os.path.join(PROC_DIR,f"{coin}.csv")
    df.to_csv(out_path)
    print(f"Saved {coin} -> {out_path}")


parquet_path=os.path.join(RAW_DIR,"ohlcv.parquet")
if os.path.exists(parquet_path):
    df=pd.read_parquet(parquet_path)
    if "date" in df.columns:
        df["date"]=pd.to_datetime(df["date"])
        df.set_index("date",inplace=True)
    df.to_csv(os.path.join(PROC_DIR,"ohlcv.csv"))
    print("Saved ohlcv -> processed/ohlcv.csv")