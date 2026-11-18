"""Download, standardize, validate, and save FRED macro data (point-in-time).

Run from the project root:
    python -m scripts.fetch_macro
"""
import logging
import sys

import pandas as pd

import config
from src.macro_data import MacroDataError, fetch_series, standardize_series, validate_macro

logger = logging.getLogger("fetch_macro")


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    frames, issues, failed = [], [], []

    for series_id, name in config.MACRO_SERIES.items():
        use_vintages = series_id in config.MACRO_VINTAGE_SERIES
        try:
            raw = fetch_series(series_id, config.MACRO_START_DATE, vintages=use_vintages)
        except MacroDataError as exc:
            logger.error("%s", exc)
            failed.append(series_id)
            continue

        raw.to_parquet(config.RAW_DIR / f"fred_{name}.parquet", index=False)

        df = standardize_series(raw, name, use_vintages=use_vintages, tz=config.STORAGE_TZ)
        max_age = config.MACRO_MAX_AGE_DAYS["monthly" if use_vintages else "daily"]
        frames.append(df)
        issues.append(validate_macro(df, max_age_days=max_age))

    # All-or-nothing: a partial macro file would silently drop indicators from scoring
    if failed:
        logger.error("Failed series %s; processed macro data NOT updated.", failed)
        return 1

    macro = pd.concat(frames, ignore_index=True)
    out_path = config.PROCESSED_DIR / "macro.parquet"
    macro.to_parquet(out_path, index=False)

    flagged = [i for i in issues if len(i)]
    if flagged:
        pd.concat(flagged, ignore_index=True).to_parquet(
            config.PROCESSED_DIR / "macro_issues.parquet", index=False
        )

    logger.info("Saved %d rows for %d indicators -> %s", len(macro), macro["indicator"].nunique(), out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())