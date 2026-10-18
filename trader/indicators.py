"""Technical indicators computed from daily OHLCV data.

Every function takes pandas Series/DataFrames indexed by date and returns a
Series aligned to the same index, so results can be dropped straight into the
price frame.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def sma(close: pd.Series, window: int) -> pd.Series:
    return close.rolling(window, min_periods=window).mean()


def ema(close: pd.Series, span: int) -> pd.Series:
    return close.ewm(span=span, adjust=False).mean()


def rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Wilder's RSI (0-100)."""
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = gain / loss.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    # No losses in the window means maximum strength.
    return out.where(loss != 0, 100.0)


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    return pd.DataFrame({"macd": line, "signal": sig, "macd_hist": line - sig})


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range using Wilder smoothing."""
    prev_close = df["Close"].shift()
    tr = pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - prev_close).abs(),
            (df["Low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def historical_volatility(close: pd.Series, window: int = 20) -> pd.Series:
    """Annualised close-to-close volatility (0.30 == 30%)."""
    log_ret = np.log(close / close.shift())
    return log_ret.rolling(window, min_periods=window).std() * np.sqrt(TRADING_DAYS)


def relative_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    """Today's volume divided by the average of the previous `window` days."""
    avg = volume.shift().rolling(window, min_periods=window).mean()
    return volume / avg.replace(0, np.nan)


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of an OHLCV frame with every indicator column attached."""
    out = df.copy()
    close = out["Close"]
    out["SMA20"] = sma(close, 20)
    out["SMA50"] = sma(close, 50)
    out["SMA200"] = sma(close, 200)
    out["RSI"] = rsi(close)
    out = out.join(macd(close))
    out["ATR"] = atr(out)
    out["HV20"] = historical_volatility(close)
    out["RelVol"] = relative_volume(out["Volume"])
    out["High20"] = out["High"].rolling(20, min_periods=20).max()
    return out
