import pandas as pd
import numpy as np
import os

def add_returns(df):
    df["return_pct"]=df["close"].pct_change(1)
    df["return_log"]=np.log(df["close"]/df["close"].shift(1))
    return df

def simple_moving_averages(df):
    for window in [7,14,30]:
        df[f"sma_{window}"]=df["close"].rolling(window).mean()
    return df

def add_volatitlity(df):
    for window in [7,14]:
        df[f"vol_{window}"]=df["return_pct"].rolling(window).std()
    return df


def add_roc(df,window=7):
    df[f"roc_{window}"]=df["close"].pct_change(periods=window)
    return df

def add_rsi(df,window=14):
    delta=df["close"].diff()
    gain=np.where(delta>0,delta,0)
    loss=np.where(delta<0,-delta,0)

    avg_gain=pd.Series(gain).rolling(window).mean()
    avg_loss=pd.Series(loss).rolling(window).mean()

    rs=avg_gain/avg_loss
    df[f"rsi_{window}"]=100-(100/(1+rs))
    return df


def add_lags(df,max_lag=7):
    for lag in range(1,max_lag+1):
        df[f"lag_{lag}"]=df["close"].shift(lag)
    return df


def save(df,path="data/processed/features.parquet"):
    df.to_parquet(path,index=False)


def build_features(df):
    df=add_returns(df)
    df=simple_moving_averages(df)
    df=add_volatitlity(df)
    df=add_roc(df)
    df=add_rsi(df)
    df=add_lags(df)
    return df