from pathlib import Path

DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


MODELS_DIR=Path("models")
MODELS_DIR.mkdir(parents=True,exist_ok=True)


DEFAULT_DAYS = 30
MAX_DAYS = 90  
VS_CURRENCY = "usd"


COIN_MAP = {
    "bitcoin": "bitcoin",
    "ethereum": "ethereum", 
    "dogecoin": "dogecoin",
    "cardano": "cardano",
    "solana": "solana",
    "matic": "matic-network",  
    "chainlink": "chainlink",
    "litecoin": "litecoin",
    "binancecoin": "binancecoin",  
    "ripple": "ripple"  
}