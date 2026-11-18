"""Central configuration for AurumAI."""
from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# --- Market data ---
# yfinance ticker -> short name used throughout the project
MARKET_SYMBOLS: dict[str, str] = {
    "GC=F": "gold",
    "DX-Y.NYB": "dxy",
    "^TNX": "us10y",
    "CL=F": "wti_oil",
    "^GSPC": "sp500",
    "BTC-USD": "bitcoin",
}

PRIMARY_SYMBOL = "GC=F"

MARKET_START_DATE = "2015-01-01"
MARKET_INTERVAL = "1d"

# All timestamps in the project are stored in this timezone
STORAGE_TZ = "UTC"