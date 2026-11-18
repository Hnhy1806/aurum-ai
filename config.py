"""Central configuration for AurumAI."""
from pathlib import Path

import os
from dotenv import load_dotenv

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

# --- Secrets (loaded from .env, never hardcoded) ---
load_dotenv(PROJECT_ROOT / ".env")
FRED_API_KEY = os.getenv("FRED_API_KEY")

# --- Macro data (FRED) ---
FRED_BASE_URL = "https://api.stlouisfed.org/fred"

# FRED series ID -> short name
MACRO_SERIES: dict[str, str] = {
    "DFII10": "real_yield_10y",
    "DGS10": "yield_10y",
    "T10YIE": "breakeven_10y",
    "DGS2": "yield_2y",
    "EFFR": "fed_funds",
    "CPIAUCSL": "cpi",
    "CPILFESL": "core_cpi",
    "PCEPILFE": "core_pce",
    "UNRATE": "unemployment",
}

# Monthly series are revised and published with a lag -> use FRED vintage history
MACRO_VINTAGE_SERIES = {"CPIAUCSL", "CPILFESL", "PCEPILFE", "UNRATE"}

# Daily series are not revised and are published the next business day
DAILY_RELEASE_LAG_BDAYS = 1

# Release times (US Eastern) used to build available_at
MONTHLY_RELEASE_TIME_ET = "08:30"  # BLS (CPI, jobs) and BEA (PCE) release at 8:30 AM ET
DAILY_RELEASE_TIME_ET = "16:30"    # Fed H.15 rates post ~4:15 PM ET the next business day

MACRO_START_DATE = MARKET_START_DATE

# Warn if a series' latest observation is older than this (FRED stopped updating, etc.)
MACRO_MAX_AGE_DAYS = {"daily": 10, "monthly": 100}