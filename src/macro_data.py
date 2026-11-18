"""Macroeconomic data collection from FRED."""
import logging

import pandas as pd
import numpy as np
from pandas.tseries.holiday import USFederalHolidayCalendar
import requests

import config

logger = logging.getLogger(__name__)

MACRO_COLUMNS = ["date", "indicator", "value", "available_at"]

class MacroDataError(Exception):
    """Raised when macro data cannot be retrieved."""


def fetch_series(series_id: str, start: str, api_key: str | None = None, vintages: bool = False) -> pd.DataFrame:
    """Download raw observations for one FRED series.

    vintages=False: latest revised values only.
    vintages=True:  every published version, with realtime_start = publication date.
    """
    api_key = api_key or config.FRED_API_KEY
    if not api_key:
        raise MacroDataError("FRED_API_KEY is not set. Add it to .env")

    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start,
    }
    if vintages:
        params["realtime_start"] = start
        params["realtime_end"] = "9999-12-31"
    
    logger.info("Downloading FRED %s (start=%s)", series_id, start)

    try:
        resp = requests.get(f"{config.FRED_BASE_URL}/series/observations", params=params, timeout=30)
    except requests.RequestException as exc:
        # Error messages can contain the full URL, including the key. Redact it.
        raise MacroDataError(f"Network error for {series_id}: {str(exc).replace(api_key, '***')}") from None

    if resp.status_code != 200:
        try:
            detail = resp.json().get("error_message", "")
        except ValueError:
            detail = resp.text[:200]
        raise MacroDataError(f"FRED {series_id}: HTTP {resp.status_code} {detail}")

    observations = resp.json().get("observations", [])
    if not observations:
        raise MacroDataError(f"No observations returned for {series_id}")

    df = pd.DataFrame(observations)
    logger.info("%s: %d observations", series_id, len(df))
    return df


_US_HOLIDAYS = (
    USFederalHolidayCalendar()
    .holidays(start="2000-01-01", end="2040-12-31")
    .values.astype("datetime64[D]")
)


def _add_business_days(days: pd.Series, n: int) -> pd.Series:
    """Shift dates forward by n US business days (skips weekends and federal holidays)."""
    shifted = np.busday_offset(days.values.astype("datetime64[D]"), n, roll="forward", holidays=_US_HOLIDAYS)
    return pd.Series(pd.to_datetime(shifted), index=days.index)


def _at_release_time(days: pd.Series, time_et: str, tz: str) -> pd.Series:
    """Combine a calendar date with a US Eastern release time, then convert to tz."""
    local = pd.to_datetime(days.dt.strftime("%Y-%m-%d") + " " + time_et)
    return local.dt.tz_localize("America/New_York").dt.tz_convert(tz)


def standardize_series(raw: pd.DataFrame, name: str, use_vintages: bool, tz: str = "UTC") -> pd.DataFrame:
    """Convert raw FRED observations to the AurumAI point-in-time macro schema.

    date         = reference period (e.g. 2026-10-01 means 'October')
    available_at = when this value was actually published
    """
    ref_date = pd.to_datetime(raw["date"])

    if use_vintages:
        published = pd.to_datetime(raw["realtime_start"])
        available_at = _at_release_time(published, config.MONTHLY_RELEASE_TIME_ET, tz)
    else:
        published = _add_business_days(ref_date, config.DAILY_RELEASE_LAG_BDAYS)
        available_at = _at_release_time(published, config.DAILY_RELEASE_TIME_ET, tz)

    df = pd.DataFrame({
        "date": ref_date.dt.tz_localize(tz),
        "indicator": name,
        "value": pd.to_numeric(raw["value"], errors="coerce"),
        "available_at": available_at,
    })

    n_missing = int(df["value"].isna().sum())
    if n_missing:
        logger.info("%s: dropped %d missing ('.') observations", name, n_missing)

    return (
        df.dropna(subset=["value"])
        .sort_values(["date", "available_at"])
        .reset_index(drop=True)[MACRO_COLUMNS]
    )

def validate_macro(df: pd.DataFrame, max_age_days: int) -> pd.DataFrame:
    """Run data-quality checks on one standardized series. Returns flagged rows; does not modify df."""
    name = df["indicator"].iloc[0] if len(df) else "?"

    checks = {
        "available_before_reference": df["available_at"] <= df["date"],
        "duplicate_version": df.duplicated(subset=["date", "available_at"], keep=False),
    }

    flagged = []
    for issue, mask in checks.items():
        n = int(mask.sum())
        if n:
            logger.warning("%s: %d rows failed '%s'", name, n, issue)
            flagged.append(df.loc[mask, ["date", "indicator"]].assign(issue=issue))

    latest = df["date"].max()
    age_days = (pd.Timestamp.now(tz="UTC") - latest).days
    if age_days > max_age_days:
        logger.warning("%s: latest observation is %d days old (limit %d)", name, age_days, max_age_days)
        flagged.append(pd.DataFrame({"date": [latest], "indicator": [name], "issue": ["stale_series"]}))

    if not flagged:
        logger.info("%s: all checks passed", name)
        return pd.DataFrame(columns=["date", "indicator", "issue"])

    return pd.concat(flagged, ignore_index=True)