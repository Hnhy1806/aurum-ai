"""Market data collection for AurumAI."""
import logging
import warnings

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore", module="yfinance")

class MarketDataError(Exception):
    """Raised when market data cannot be retrieved."""


# a function that downloads the raw price history for one ticker and nothing else
def fetch_symbol(
    ticker: str,
    start: str,
    end: str | None = None,
    interval: str = "1d",
) -> pd.DataFrame:
    """Download raw OHLCV history for one ticker from yfinance."""
    logger.info("Downloading %s (interval=%s, start=%s)", ticker, interval, start)

    try:
        df = yf.download(
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=False,
            multi_level_index=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        raise MarketDataError(f"Download failed for {ticker}: {exc}") from exc

    if df is None or df.empty:
        raise MarketDataError(f"No data returned for {ticker}")

    logger.info(
        "%s: %d rows, %s to %s",
        ticker, len(df), df.index.min().date(), df.index.max().date(),
    )
    return df

# a function that turns yfinance’s raw output into one fixed shape that the rest of AurumAI can rely on
STANDARD_COLUMNS = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]
PRICE_COLUMNS = ["open", "high", "low", "close"]


def standardize(raw: pd.DataFrame, symbol: str, tz: str = "UTC") -> pd.DataFrame:
    """Convert raw yfinance output to the AurumAI standard schema (long format)."""
    df = raw.rename(columns=str.lower)

    missing = set(PRICE_COLUMNS + ["volume"]) - set(df.columns)
    if missing:
        raise MarketDataError(f"{symbol}: missing columns {sorted(missing)}")

    # The index is named 'Date' for daily data and 'Datetime' for intraday
    df = df.reset_index()
    df = df.rename(columns={df.columns[0]: "timestamp"})

    ts = pd.to_datetime(df["timestamp"])
    ts = ts.dt.tz_localize(tz) if ts.dt.tz is None else ts.dt.tz_convert(tz)
    df["timestamp"] = ts

    df["symbol"] = symbol
    df[PRICE_COLUMNS + ["volume"]] = df[PRICE_COLUMNS + ["volume"]].astype("float64")

    return (
        df[STANDARD_COLUMNS]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

# a function that runs data-quality checks on the standardized data and reports the problems it finds
def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Run data-quality checks. Returns one row per flagged record; does not modify df."""
    symbol = df["symbol"].iloc[0] if len(df) else "?"

    checks = {
        "missing_price": df[PRICE_COLUMNS].isna().any(axis=1),
        "duplicate_timestamp": df["timestamp"].duplicated(keep=False),
        "high_below_low": df["high"] < df["low"],
        "close_outside_range": (df["close"] > df["high"]) | (df["close"] < df["low"]),
        "non_positive_price": (df[PRICE_COLUMNS] <= 0).any(axis=1),
        "repeated_volume": (df["volume"] > 0) & (df["volume"] == df["volume"].shift(1)),
    }

    flagged = []
    for issue, mask in checks.items():
        n = int(mask.sum())
        if n:
            logger.warning("%s: %d rows failed '%s'", symbol, n, issue)
            flagged.append(df.loc[mask, ["timestamp", "symbol"]].assign(issue=issue))

    if not flagged:
        logger.info("%s: all checks passed", symbol)
        return pd.DataFrame(columns=["timestamp", "symbol", "issue"])

    return pd.concat(flagged, ignore_index=True).sort_values("timestamp")