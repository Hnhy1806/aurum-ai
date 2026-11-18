"""Download, standardize, validate, and save daily market data.

Run from the project root:
    python -m scripts.fetch_market
"""
import logging
import sys

import pandas as pd

import config
from src.market_data import MarketDataError, fetch_symbol, standardize, validate

logger = logging.getLogger("fetch_market")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    interval = config.MARKET_INTERVAL

    frames, issues, failed = [], [], []

    for ticker, name in config.MARKET_SYMBOLS.items():
        try:
            raw = fetch_symbol(ticker, start=config.MARKET_START_DATE, interval=interval)
        except MarketDataError as exc:
            logger.error("%s", exc)
            failed.append(ticker)
            continue

        # Save the untouched download first
        raw.reset_index().to_parquet(config.RAW_DIR / f"{name}_{interval}.parquet", index=False)

        df = standardize(raw, name, tz=config.STORAGE_TZ)
        frames.append(df)
        issues.append(validate(df))

    if config.PRIMARY_SYMBOL in failed:
        logger.error("Primary symbol %s failed; processed data NOT updated.", config.PRIMARY_SYMBOL)
        return 1

    market = pd.concat(frames, ignore_index=True)
    out_path = config.PROCESSED_DIR / f"market_{interval}.parquet"
    market.to_parquet(out_path, index=False)

    flagged = [i for i in issues if len(i)]
    if flagged:
        pd.concat(flagged, ignore_index=True).to_parquet(
            config.PROCESSED_DIR / f"market_{interval}_issues.parquet", index=False
        )

    logger.info("Saved %d rows for %d symbols -> %s", len(market), market["symbol"].nunique(), out_path)
    if failed:
        logger.warning("Skipped symbols: %s", failed)
    return 0


if __name__ == "__main__":
    sys.exit(main())