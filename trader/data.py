"""Market data: live quotes from Yahoo Finance (via yfinance) or offline demo data.

Yahoo data is free but delayed ~15 minutes and occasionally patchy, especially
option quotes outside market hours. Always confirm the live bid/ask in
Robinhood before placing an order.
"""

from __future__ import annotations

import zlib
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from .options import bs_price

DEFAULT_WATCHLIST = [
    "SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMD", "TSLA", "META",
    "AMZN", "GOOGL", "NFLX", "AVGO", "PLTR", "COIN", "JPM", "UBER",
]


# --------------------------------------------------------------------------- live


def fetch_history(tickers: list[str], period: str = "1y") -> dict[str, pd.DataFrame]:
    import yfinance as yf

    raw = yf.download(
        tickers, period=period, interval="1d", group_by="ticker",
        auto_adjust=True, progress=False, threads=True,
    )
    out: dict[str, pd.DataFrame] = {}
    for t in tickers:
        try:
            df = raw[t] if isinstance(raw.columns, pd.MultiIndex) else raw
        except KeyError:
            continue
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        if not df.empty:
            out[t] = df
    return out


def fetch_option_chains(ticker: str, max_dte: int = 60, max_expiries: int = 6) -> dict[date, dict[str, pd.DataFrame]]:
    import yfinance as yf

    tk = yf.Ticker(ticker)
    today = date.today()
    chains: dict[date, dict[str, pd.DataFrame]] = {}
    for exp in tk.options:
        expiry = datetime.strptime(exp, "%Y-%m-%d").date()
        if (expiry - today).days > max_dte:
            break
        oc = tk.option_chain(exp)
        chains[expiry] = {"calls": oc.calls, "puts": oc.puts}
        if len(chains) >= max_expiries:
            break
    return chains


# --------------------------------------------------------------------------- demo


def _rng(ticker: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(ticker.encode()))


def demo_history(ticker: str, days: int = 260, end: date | None = None) -> pd.DataFrame:
    """Deterministic synthetic OHLCV so the dashboard works with no network."""
    rng = _rng(ticker)
    end = end or date.today()
    idx = pd.bdate_range(end=end, periods=days)
    drift = rng.uniform(-0.0015, 0.0025)
    vol = rng.uniform(0.012, 0.035)
    rets = rng.normal(drift, vol, days)
    # Give a few names a recent momentum burst so the ranking has variety.
    rets[-15:] += rng.uniform(-0.004, 0.006)
    close = rng.uniform(20, 500) * np.exp(np.cumsum(rets))
    open_ = close * (1 + rng.normal(0, vol / 3, days))
    high = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, vol / 2, days)))
    low = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, vol / 2, days)))
    base_vol = rng.uniform(5e6, 6e7)
    volume = base_vol * np.exp(rng.normal(0, 0.3, days)) * (1 + 3 * np.abs(rets))
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume.astype(int)},
        index=idx,
    )


def demo_option_chains(ticker: str, spot: float, hv: float, today: date | None = None) -> dict[date, dict[str, pd.DataFrame]]:
    rng = _rng(ticker + "opt")
    today = today or date.today()
    # Weekly Friday expirations for the next ~2 months.
    first_friday = today + timedelta(days=(4 - today.weekday()) % 7 or 7)
    expiries = [first_friday + timedelta(weeks=w) for w in range(9)]
    step = 1 if spot < 50 else 2.5 if spot < 150 else 5
    strikes = np.arange(round(spot * 0.75 / step) * step, spot * 1.25, step)
    chains = {}
    for exp in expiries:
        t = (exp - today).days / 365
        sides = {}
        for kind in ("call", "put"):
            rows = []
            for k in strikes:
                moneyness = np.log(k / spot)
                iv = max(0.08, hv * rng.uniform(0.95, 1.35) + 0.25 * moneyness**2 - 0.1 * moneyness)
                fair = bs_price(spot, k, t, iv, kind)
                if fair < 0.05:
                    continue
                half_spread = max(0.01, fair * rng.uniform(0.01, 0.08))
                oi = int(rng.lognormal(7, 1.2) * np.exp(-8 * abs(moneyness)))
                rows.append({
                    "contractSymbol": f"{ticker}{exp:%y%m%d}{kind[0].upper()}{int(k * 1000):08d}",
                    "strike": float(k),
                    "lastPrice": round(fair, 2),
                    "bid": round(fair - half_spread, 2),
                    "ask": round(fair + half_spread, 2),
                    "volume": int(oi * rng.uniform(0.02, 0.4)),
                    "openInterest": oi,
                    "impliedVolatility": iv,
                })
            sides[kind + "s"] = pd.DataFrame(rows)
        chains[exp] = sides
    return chains
