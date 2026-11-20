"""
Incremental news collection for AurumAI.

Each run:
  1. loads the stored history (data/news.csv),
  2. fetches the latest news (Finnhub + Fed RSS; GDELT with --gdelt),
  3. appends only articles whose article_id is not already stored,
  4. saves atomically and logs a one-line summary.

Safe to re-run any time: duplicates are skipped by article_id.

Usage (from the project root):
    python scripts/fetch_news.py
    python scripts/fetch_news.py --gdelt      # also query GDELT (slow, rate-limited)
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Make `src` importable no matter which folder the script is started from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.news import NEWS_COLUMNS, collect_news, validate_news  # noqa: E402

NEWS_PATH = PROJECT_ROOT / "data" / "news.csv"
LOG_PATH = PROJECT_ROOT / "logs" / "fetch_news.log"

logger = logging.getLogger("fetch_news")


def setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler()],
    )


def load_history(path: Path) -> pd.DataFrame:
    """Load stored news, or an empty frame on the first run."""
    if not path.exists():
        logger.info("No history at %s yet; starting fresh", path)
        return pd.DataFrame(columns=NEWS_COLUMNS)
    df = pd.read_csv(path, dtype={"article_id": str})
    for col in ("published_at", "fetched_at"):
        df[col] = pd.to_datetime(df[col], utc=True)  # CSV stores text; restore UTC timestamps
    return df


def save_atomic(df: pd.DataFrame, path: Path) -> None:
    """Write to a temp file, then rename. An interrupted run can't leave a half-written CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Append new news articles to data/news.csv")
    parser.add_argument("--gdelt", action="store_true", help="also query GDELT (slow)")
    args = parser.parse_args()

    setup_logging()
    history = load_history(NEWS_PATH)

    latest = collect_news(include_gdelt=args.gdelt)
    if latest.empty:
        logger.error("Run failed: no articles from any source. History unchanged (%d rows).", len(history))
        return 1  # non-zero exit code tells a scheduler the run failed

    new = latest[~latest["article_id"].isin(history["article_id"])]
    # On the first run history is an empty, untyped frame; concatenating it would
    # turn the timestamp columns into plain objects, so skip the concat then.
    combined = new if history.empty else pd.concat([history, new], ignore_index=True)
    combined = combined.sort_values("published_at", kind="stable").reset_index(drop=True)

    issues = validate_news(combined)
    if any(issues.values()):
        logger.warning("Validation issues after merge: %s", issues)

    save_atomic(combined[NEWS_COLUMNS], NEWS_PATH)
    logger.info("Run OK: fetched %d, new %d, total stored %d (newest article %s)",
                len(latest), len(new), len(combined), combined["published_at"].max())
    return 0


if __name__ == "__main__":
    sys.exit(main())