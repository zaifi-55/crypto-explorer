import os
import requests
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
import time
import streamlit as st
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import RAW_DIR, PROCESSED_DIR, DEFAULT_DAYS, MAX_DAYS, VS_CURRENCY, COIN_MAP

load_dotenv()
API_KEY = os.getenv("CC_M")  

# Rate limiting configuration
REQUESTS_PER_MINUTE = 10  # CoinGecko free tier allows ~10-50 calls per minute
REQUEST_DELAY = 6  # 6 seconds between requests to stay under limit

def create_session_with_retries():
    """Create a requests session with retry strategy"""
    session = requests.Session()
    
    # Define retry strategy
    retry_strategy = Retry(
        total=3,  # Total number of retries
        backoff_factor=2,  # Wait time between retries: 2, 4, 8 seconds
        status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry
        allowed_methods=["GET"]  # Only retry GET requests
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

# Global session instance
SESSION = create_session_with_retries()

def validate_coin_id(coin_symbol: str) -> str:
    """Validate and return the correct CoinGecko coin ID"""
    if coin_symbol not in COIN_MAP:
        st.error(f"Unsupported coin: {coin_symbol}")
        st.info(f"Available coins: {', '.join(COIN_MAP.keys())}")
        return None
    
    return COIN_MAP[coin_symbol]

def test_coin_availability(coin_id: str) -> bool:
    """Test if a coin ID is available in CoinGecko API"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}"
        r = SESSION.get(url, timeout=10)
        return r.status_code == 200
    except:
        return False  


def fetch_price_history(coin: str, days: int = 90, vs_currency: str = "usd") -> pd.DataFrame:
    """Fetch price history with proper error handling and rate limiting"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin}/market_chart"
        params = {"vs_currency": vs_currency, "days": days}
        
        # Add rate limiting with progress indicator
        with st.spinner(f"Fetching price data for {coin}... (Rate limited for API stability)"):
            time.sleep(REQUEST_DELAY)  # Rate limiting
        
        # Make request with session and timeout
        r = SESSION.get(url, params=params, timeout=30)
        
        # Check for rate limiting specifically
        if r.status_code == 429:
            st.error("⚠️ API Rate limit reached. Please wait a moment and try again.")
            st.info("💡 CoinGecko limits free API calls. Try waiting 1-2 minutes before switching coins.")
            return pd.DataFrame(columns=["timestamp", "price", "date"])
        
        r.raise_for_status()  # Raise an exception for other bad status codes
        
        data = r.json()
        
        if "prices" not in data:
            st.error(f"No price data available for {coin}")
            return pd.DataFrame(columns=["timestamp", "price", "date"])
        
        prices = data["prices"]
        if not prices:
            st.warning(f"Empty price data for {coin}")
            return pd.DataFrame(columns=["timestamp", "price", "date"])
            
        df = pd.DataFrame(prices, columns=["timestamp", "price"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        return df
        
    except requests.exceptions.Timeout:
        st.error(f"⏱️ Request timed out for {coin}. Please try again.")
        return pd.DataFrame(columns=["timestamp", "price", "date"])
    except requests.exceptions.ConnectionError:
        st.error(f"🌐 Connection error. Please check your internet connection.")
        return pd.DataFrame(columns=["timestamp", "price", "date"])
    except requests.exceptions.RequestException as e:
        if "429" in str(e):
            st.error("🚫 Too many requests. Please wait before trying again.")
            st.info("💡 Tip: Wait 1-2 minutes between coin switches to avoid rate limits.")
        else:
            st.error(f"Error fetching price data for {coin}: {str(e)}")
        return pd.DataFrame(columns=["timestamp", "price", "date"])
    except Exception as e:
        st.error(f"Unexpected error fetching price data: {str(e)}")
        return pd.DataFrame(columns=["timestamp", "price", "date"])


def cache_data(coin: str, days: int = 90) -> pd.DataFrame:
    """Cache data with improved error handling and longer cache validity"""
    try:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        cache_file = RAW_DIR / f"{coin}_{days}d.csv"
        
        # Check if cache exists and is recent (less than 30 minutes old for rate limiting)
        if cache_file.exists():
            cache_age = time.time() - cache_file.stat().st_mtime
            if cache_age < 1800:  # 30 minutes in seconds (longer cache to avoid rate limits)
                try:
                    df = pd.read_csv(cache_file)
                    df["date"] = pd.to_datetime(df["date"]).dt.date
                    st.success(f"✅ Using cached data for {coin} (cached {int(cache_age/60)} minutes ago)")
                    return df
                except Exception as e:
                    st.warning(f"Error reading cache file, fetching fresh data: {str(e)}")
        
        # Fetch fresh data
        df = fetch_price_history(coin, days)
        
        if not df.empty:
            df.to_csv(cache_file, index=False)
            st.success(f"✅ Fresh data fetched and cached for {coin}")
        
        return df
        
    except Exception as e:
        st.error(f"Error in cache_data: {str(e)}")
        return pd.DataFrame(columns=["timestamp", "price", "date"])


def fetch_ohlc(coin_id: str, days: int, vs_currency: str = VS_CURRENCY) -> pd.DataFrame:
    """Fetch OHLC data with proper error handling and API limitations"""
    try:
        # CoinGecko's free API only supports OHLC data for up to 90 days
        if days > 90:
            st.warning(f"⚠️ OHLC data is only available for up to 90 days. Requested: {days} days")
            return pd.DataFrame(columns=["date", "open", "high", "low", "close"])
            
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/ohlc"
        params = {"vs_currency": vs_currency, "days": days}
        
        # Add rate limiting with progress indicator
        with st.spinner(f"Fetching OHLC data for {coin_id}..."):
            time.sleep(REQUEST_DELAY)  # Rate limiting
        
        r = SESSION.get(url, params=params, timeout=30)
        
        # Check for rate limiting specifically
        if r.status_code == 429:
            st.error("⚠️ API Rate limit reached for OHLC data. Using cached data if available.")
            return pd.DataFrame(columns=["date", "open", "high", "low", "close"])
        
        r.raise_for_status()
        
        data = r.json()
        
        if not data or not isinstance(data, list):
            st.warning(f"No OHLC data available for {coin_id} (days: {days})")
            return pd.DataFrame(columns=["date", "open", "high", "low", "close"])
        
        if len(data) == 0:
            st.warning(f"Empty OHLC data returned for {coin_id}")
            return pd.DataFrame(columns=["date", "open", "high", "low", "close"])
        
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close"])
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.date
        return df[["date", "open", "high", "low", "close"]]
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching OHLC data for {coin_id}: {str(e)}")
        return pd.DataFrame(columns=["date", "open", "high", "low", "close"])
    except Exception as e:
        st.error(f"Unexpected error fetching OHLC data: {str(e)}")
        return pd.DataFrame(columns=["date", "open", "high", "low", "close"])


def fetch_daily_volume(coin_id: str, days: int, vs_currency: str = VS_CURRENCY) -> pd.DataFrame:
    """Fetch volume data with proper error handling"""
    try:
        url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
        params = {"vs_currency": vs_currency, "days": days}
        
        # Add rate limiting
        time.sleep(REQUEST_DELAY)
        
        r = SESSION.get(url, params=params, timeout=30)
        
        # Check for rate limiting
        if r.status_code == 429:
            st.warning("⚠️ Rate limit reached for volume data. Proceeding without volume.")
            return pd.DataFrame(columns=["date", "volume"])
        
        r.raise_for_status()
        
        data = r.json()
        
        if "total_volumes" not in data:
            st.warning(f"No volume data available for {coin_id}")
            return pd.DataFrame(columns=["date", "volume"])
        
        vols = data.get("total_volumes", [])
        
        if not vols:
            st.warning(f"Empty volume data for {coin_id}")
            return pd.DataFrame(columns=["date", "volume"])
        
        vdf = pd.DataFrame(vols, columns=["timestamp", "volume"])
        vdf["date"] = pd.to_datetime(vdf["timestamp"], unit="ms").dt.date
        vdf = vdf.groupby("date", as_index=False)["volume"].sum()
        return vdf
        
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching volume data for {coin_id}: {str(e)}")
        return pd.DataFrame(columns=["date", "volume"])
    except Exception as e:
        st.error(f"Unexpected error fetching volume data: {str(e)}")
        return pd.DataFrame(columns=["date", "volume"])


def build_ohlcv(coin_symbol: str, days: int = DEFAULT_DAYS) -> pd.DataFrame:
    """Build OHLCV data with improved error handling"""
    try:
        if coin_symbol not in COIN_MAP:
            st.error(f"Unknown coin symbol: {coin_symbol}")
            return pd.DataFrame(columns=["date", "coin", "open", "high", "low", "close", "volume"])
        
        coin_id = COIN_MAP[coin_symbol]
        
        # Fetch OHLC and volume data
        ohlc = fetch_ohlc(coin_id, days)
        vol = fetch_daily_volume(coin_id, days)
        
        # Handle empty dataframes
        if ohlc.empty and vol.empty:
            st.error(f"No data available for {coin_symbol}")
            return pd.DataFrame(columns=["date", "coin", "open", "high", "low", "close", "volume"])
        
        if ohlc.empty:
            st.warning(f"No OHLC data available for {coin_symbol}")
            return pd.DataFrame(columns=["date", "coin", "open", "high", "low", "close", "volume"])
        
        if vol.empty:
            st.warning(f"No volume data available for {coin_symbol}, proceeding without volume")
            df = ohlc.copy()
            df["volume"] = 0  # Add empty volume column
        else:
            # Merge OHLC and volume data
            df = pd.merge(ohlc, vol, on="date", how="left")
            df["volume"] = df["volume"].fillna(0)  # Fill missing volumes with 0
        
        df["coin"] = coin_symbol
        
        # Ensure proper column order
        return df[["date", "coin", "open", "high", "low", "close", "volume"]]
        
    except Exception as e:
        st.error(f"Error building OHLCV data for {coin_symbol}: {str(e)}")
        return pd.DataFrame(columns=["date", "coin", "open", "high", "low", "close", "volume"])


def cache_raw_csv(coin_symbol: str, days: int = DEFAULT_DAYS, force: bool = False) -> Path:
    """Cache raw CSV with error handling"""
    try:
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        path = RAW_DIR / f"{coin_symbol}.csv"
        
        if path.exists() and not force:
            return path
        
        df = build_ohlcv(coin_symbol, days)
        
        if not df.empty:
            df.to_csv(path, index=False)
        
        return path
        
    except Exception as e:
        st.error(f"Error caching CSV for {coin_symbol}: {str(e)}")
        return None


def build_processed_parquet(outname: str = "prices.parquet") -> Path:
    """Build processed parquet with error handling"""
    try:
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        all_dfs = []
        
        for sym in COIN_MAP.keys():
            f = RAW_DIR / f"{sym}.csv"
            if f.exists():
                try:
                    df = pd.read_csv(f)
                    df["date"] = pd.to_datetime(df["date"]).dt.date
                    all_dfs.append(df)
                except Exception as e:
                    st.warning(f"Error reading {f}: {str(e)}")
        
        if not all_dfs:
            st.warning("No data files found to process")
            return None
        
        df = pd.concat(all_dfs, ignore_index=True)
        out = PROCESSED_DIR / outname
        df.to_parquet(out, index=False)
        return out
        
    except Exception as e:
        st.error(f"Error building processed parquet: {str(e)}")
        return None


if __name__ == "__main__":
    for coin in ["bitcoin", "ethereum", "dogecoin"]:
        try:
            df = cache_data(coin, days=365)
            print(f"{coin} -> {len(df)} rows")
            if not df.empty:
                print(df.head(), "\n")
            else:
                print("No data retrieved\n")
        except Exception as e:
            print(f"Error processing {coin}: {str(e)}\n")