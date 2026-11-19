"""Build the point-in-time daily feature table for gold."""
import pandas as pd
import numpy as np

import config
from src import indicators as ind
from src.market_data import load_symbol

CROSS_ASSETS = {"dxy": False, "us10y": True, "wti_oil": False, "sp500": False, "bitcoin": False}  # name -> is_yield
ASOF_TOLERANCE = pd.Timedelta(days=5)


def bar_available_at(index: pd.DatetimeIndex, symbol: str) -> pd.DatetimeIndex:
    """UTC time at which a daily bar dated D is final."""
    if symbol in config.MARKET_24_7_SYMBOLS:
        avail = index + pd.Timedelta(days=1)  # UTC day D ends at 00:00 UTC on D+1
    else:
        local = pd.to_datetime(index.strftime("%Y-%m-%d") + " " + config.MARKET_CLOSE_TIME_ET)
        avail = local.tz_localize("America/New_York").tz_convert("UTC")
    return avail.as_unit("us")


def gold_technical_features() -> pd.DataFrame:
    """Scale-free technical features for gold, one row per session."""
    gold = load_symbol("gold")
    c = gold["close"]
    a = ind.atr(gold)
    m = ind.macd(c)

    f = pd.DataFrame(index=gold.index)
    f["gold_close"] = c
    f["ret_1d"] = ind.simple_returns(c)
    f["mom_21d"] = ind.simple_returns(c, 21)
    f["mom_63d"] = ind.simple_returns(c, 63)
    f["pct_from_ema20"] = c / ind.ema(c, 20) - 1
    f["pct_from_sma50"] = c / ind.sma(c, 50) - 1
    f["pct_from_sma200"] = c / ind.sma(c, 200) - 1
    f["rsi_14"] = ind.rsi(c)
    f["macd_pct"] = m["macd"] / c
    f["macd_hist_pct"] = m["macd_hist"] / c
    f["atr_pct"] = a / c
    f["rv_20d"] = ind.realized_volatility(c)
    f["dist_20d_high_atr"] = (ind.rolling_high(gold, 20) - c) / a
    f["dist_20d_low_atr"] = (c - ind.rolling_low(gold, 20)) / a
    f["drawdown"] = ind.drawdown(c)
    f["as_of"] = bar_available_at(gold.index, "gold")
    return f.reset_index().rename(columns={"timestamp": "date"})


def cross_asset_features(symbol: str, is_yield: bool) -> pd.DataFrame:
    """Close plus 1-day and 21-bar changes for one related asset, keyed by availability time."""
    df = load_symbol(symbol)
    c = df["close"]
    out = pd.DataFrame({f"{symbol}_close": c})
    if is_yield:
        out[f"{symbol}_chg_1d"] = c.diff()
        out[f"{symbol}_chg_21d"] = c.diff(21)
    else:
        out[f"{symbol}_ret_1d"] = ind.simple_returns(c)
        out[f"{symbol}_ret_21d"] = ind.simple_returns(c, 21)
    out[f"{symbol}_available_at"] = bar_available_at(df.index, symbol)
    return out.reset_index(drop=True).sort_values(f"{symbol}_available_at")


def build_market_features() -> pd.DataFrame:
    """Gold technicals joined with related assets, using only bars available at each gold close."""
    table = gold_technical_features().sort_values("as_of")

    for symbol, is_yield in CROSS_ASSETS.items():
        other = cross_asset_features(symbol, is_yield)
        avail_col = f"{symbol}_available_at"
        table = pd.merge_asof(
            table, other,
            left_on="as_of", right_on=avail_col,
            direction="backward", tolerance=ASOF_TOLERANCE,
        )
        table[f"{symbol}_stale_days"] = (table["as_of"] - table[avail_col]).dt.total_seconds() / 86400
        table = table.drop(columns=avail_col)

    return table.reset_index(drop=True)

DAILY_MACRO = ["real_yield_10y", "breakeven_10y", "yield_10y", "yield_2y", "fed_funds"]
INFLATION_MACRO = ["cpi", "core_cpi", "core_pce"]
DAILY_MACRO_TOLERANCE = pd.Timedelta(days=5)
MONTHLY_MACRO_TOLERANCE = pd.Timedelta(days=100)


def latest_known(macro: pd.DataFrame, name: str) -> pd.DataFrame:
    """For one indicator: rows that were the newest reference period when published.

    Drops later revisions of older periods, so an as-of lookup returns the current reading.
    """
    m = macro[macro["indicator"] == name].sort_values(["available_at", "date"])
    return m.loc[m["date"] == m["date"].cummax(), ["date", "value", "available_at"]].reset_index(drop=True)


def _value_months_ago(latest: pd.DataFrame, months: int) -> pd.Series:
    """First-published value of the reference period `months` earlier (NaN if never published)."""
    first_release = latest.drop_duplicates("date", keep="first").set_index("date")["value"]
    return (latest["date"] - pd.DateOffset(months=months)).map(first_release)


def _macro_frame(macro: pd.DataFrame, name: str) -> pd.DataFrame:
    """Features for one indicator, keyed by publication time."""
    lk = latest_known(macro, name)
    out = pd.DataFrame({f"{name}_available_at": lk["available_at"]})

    if name in DAILY_MACRO:
        out[name] = lk["value"]
        out[f"{name}_chg_21d"] = lk["value"].diff(21)
    else:
        out[f"{name}_ref_date"] = lk["date"]
        if name in INFLATION_MACRO:
            out[f"{name}_yoy"] = lk["value"] / _value_months_ago(lk, 12) - 1
            out[f"{name}_ann_3m"] = (lk["value"] / _value_months_ago(lk, 3)) ** 4 - 1
        else:  # unemployment
            out[name] = lk["value"]
            out[f"{name}_chg_3m"] = lk["value"] - _value_months_ago(lk, 3)
    return out.sort_values(f"{name}_available_at")


def build_feature_table() -> pd.DataFrame:
    """Full point-in-time feature table: gold technicals + related assets + macro."""
    table = build_market_features().sort_values("as_of")
    macro = pd.read_parquet(config.PROCESSED_DIR / "macro.parquet")

    for name in macro["indicator"].unique():
        frame = _macro_frame(macro, name)
        avail_col = f"{name}_available_at"
        is_daily = name in DAILY_MACRO
        table = pd.merge_asof(
            table, frame,
            left_on="as_of", right_on=avail_col,
            direction="backward",
            tolerance=DAILY_MACRO_TOLERANCE if is_daily else MONTHLY_MACRO_TOLERANCE,
        )
        if not is_daily:
            table[f"{name}_days_since_release"] = (table["as_of"] - table[avail_col]).dt.days
        table = table.drop(columns=avail_col)

    table["yield_curve_10y2y"] = table["yield_10y"] - table["yield_2y"]
    return table.reset_index(drop=True)