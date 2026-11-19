"""Build and save the point-in-time daily feature table.

Run from the project root (after fetch_market and fetch_macro):
    python -m scripts.build_features
"""
import logging
import sys

import config
from src.features import build_feature_table

logger = logging.getLogger("build_features")


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    table = build_feature_table()
    out_path = config.PROCESSED_DIR / "features_1d.parquet"
    table.to_parquet(out_path, index=False)
    logger.info("Saved %d rows x %d columns -> %s", len(table), table.shape[1], out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())