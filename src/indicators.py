"""Technical indicators for AurumAI, implemented manually with pandas."""
import numpy as np
import pandas as pd


def simple_returns(close: pd.Series, periods: int = 1) -> pd.Series:
    """Percentage change over `periods` bars: P_t / P_{t-periods} - 1."""
    return close.pct_change(periods)


def log_returns(close: pd.Series, periods: int = 1) -> pd.Series:
    """Log return over `periods` bars. NaN where either price is non-positive."""
    ratio = close / close.shift(periods)
    return np.log(ratio.where(ratio > 0))

def sma(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average over `window` bars."""
    return series.rolling(window, min_periods=window).mean()

def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average with alpha = 2 / (span + 1)."""
    return series.ewm(span=span, adjust=False, min_periods=span).mean()

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index using Wilder's smoothing (alpha = 1/period). Range 0-100."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()

    rs = avg_gain / avg_loss
    return 100 - 100 / (1 + rs)

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """MACD line, signal line, and histogram (all in price units)."""
    macd_line = ema(close, fast) - ema(close, slow)
    signal_line = ema(macd_line, signal)
    return pd.DataFrame({
        "macd": macd_line,
        "macd_signal": signal_line,
        "macd_hist": macd_line - signal_line,
    })

def true_range(df: pd.DataFrame) -> pd.Series:
    """Largest of: high-low, |high - prev close|, |low - prev close|."""
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range using Wilder's smoothing (price units)."""
    return true_range(df).ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def realized_volatility(close: pd.Series, window: int = 20, periods_per_year: int = 252) -> pd.Series:
    """Rolling standard deviation of log returns, annualized."""
    return log_returns(close).rolling(window, min_periods=window).std() * np.sqrt(periods_per_year)

def rolling_high(df: pd.DataFrame, window: int) -> pd.Series:
    """Highest high over the last `window` bars (includes the current bar)."""
    return df["high"].rolling(window, min_periods=window).max()


def rolling_low(df: pd.DataFrame, window: int) -> pd.Series:
    """Lowest low over the last `window` bars (includes the current bar)."""
    return df["low"].rolling(window, min_periods=window).min()


def drawdown(close: pd.Series) -> pd.Series:
    """Percent below the running maximum close (0 at new highs, negative otherwise)."""
    return close / close.cummax() - 1