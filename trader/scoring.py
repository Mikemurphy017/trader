"""Rating algorithm for stocks.

Each stock gets a 0-100 score built from six weighted components, each of
which is first normalised to 0-1:

    Trend       25%  price vs SMA20/50/200 and SMA50 slope
    Momentum    25%  RSI sweet spot + MACD histogram direction
    Returns     15%  5-day and 20-day performance
    Volume      15%  relative volume, rewarded only on up days
    Breakout    10%  distance from the 20-day high
    Volatility  10%  ATR% in a range that is tradeable intraday

The score measures how bullish the current technical setup is. It is not a
guarantee that the price will rise.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .indicators import add_indicators

STOCK_WEIGHTS = {
    "trend": 0.25,
    "momentum": 0.25,
    "returns": 0.15,
    "volume": 0.15,
    "breakout": 0.10,
    "volatility": 0.10,
}

MIN_HISTORY = 60


def _clip01(x: float) -> float:
    if x is None or not np.isfinite(x):
        return 0.0
    return float(min(1.0, max(0.0, x)))


def _ramp(x: float, lo: float, hi: float) -> float:
    """Linear 0 at `lo` up to 1 at `hi`."""
    return _clip01((x - lo) / (hi - lo))


def rating_label(score: float) -> str:
    if score >= 80:
        return "Strong Buy"
    if score >= 65:
        return "Buy"
    if score >= 50:
        return "Watch"
    if score >= 40:
        return "Neutral"
    return "Avoid / Bearish"


@dataclass
class StockRating:
    ticker: str
    score: float
    rating: str
    price: float
    change_pct: float
    rsi: float
    rel_volume: float
    atr: float
    hv20: float
    target: float
    stop: float
    components: dict[str, float] = field(default_factory=dict)
    reasons: list[str] = field(default_factory=list)

    @property
    def bias(self) -> str:
        if self.score >= 60:
            return "bullish"
        if self.score < 40:
            return "bearish"
        return "neutral"


def _trend(last: pd.Series, sma50_slope: float, reasons: list[str]) -> float:
    pts = 0.0
    if last.Close > last.SMA20:
        pts += 0.25
    if last.SMA20 > last.SMA50:
        pts += 0.25
        reasons.append("Short-term trend above medium-term (SMA20 > SMA50)")
    if np.isfinite(last.SMA200):
        if last.Close > last.SMA200:
            pts += 0.25
        else:
            reasons.append("Below the 200-day average (long-term downtrend)")
    else:
        pts += 0.125  # not enough history to judge; half credit
    pts += 0.25 * _ramp(sma50_slope, -0.01, 0.03)
    return _clip01(pts)


def _momentum(last: pd.Series, prev: pd.Series, reasons: list[str]) -> float:
    r = last.RSI
    if 50 <= r <= 68:
        rsi_pts = 1.0
        reasons.append(f"RSI {r:.0f}: healthy bullish momentum")
    elif 68 < r <= 78:
        rsi_pts = 1.0 - (r - 68) / 20  # starting to get stretched
        reasons.append(f"RSI {r:.0f}: strong but getting overbought")
    elif r > 78:
        rsi_pts = 0.3
        reasons.append(f"RSI {r:.0f}: overbought, pullback risk")
    else:
        rsi_pts = _ramp(r, 30, 50)
    hist_pts = 0.0
    if last["macd_hist"] > 0:
        hist_pts += 0.5
    if last["macd_hist"] > prev["macd_hist"]:
        hist_pts += 0.5
    if last["macd_hist"] > 0 and prev["macd_hist"] <= 0:
        reasons.append("Fresh MACD bullish crossover")
    return _clip01(0.55 * rsi_pts + 0.45 * hist_pts)


def rate_stock(ticker: str, history: pd.DataFrame) -> StockRating | None:
    """Score one ticker from its daily OHLCV history. Returns None if too short."""
    if history is None or len(history) < MIN_HISTORY:
        return None
    df = add_indicators(history).dropna(subset=["SMA50", "RSI", "ATR", "HV20"])
    if len(df) < 6:
        return None
    last, prev = df.iloc[-1], df.iloc[-2]
    close = df["Close"]
    reasons: list[str] = []

    sma50_slope = df["SMA50"].iloc[-1] / df["SMA50"].iloc[-6] - 1
    ret5 = close.iloc[-1] / close.iloc[-6] - 1
    ret20 = close.iloc[-1] / close.iloc[-21] - 1 if len(close) > 21 else ret5
    change = close.iloc[-1] / close.iloc[-2] - 1
    atr_pct = last.ATR / last.Close
    rel_vol = last.RelVol if np.isfinite(last.RelVol) else 1.0

    comps = {
        "trend": _trend(last, sma50_slope, reasons),
        "momentum": _momentum(last, prev, reasons),
        "returns": 0.5 * _ramp(ret5, -0.03, 0.06) + 0.5 * _ramp(ret20, -0.05, 0.15),
    }

    up_day = change > 0
    comps["volume"] = _ramp(rel_vol, 0.8, 2.0) if up_day else 0.25 * _ramp(rel_vol, 0.8, 2.0)
    if up_day and rel_vol >= 1.5:
        reasons.append(f"Buyers stepping in: {rel_vol:.1f}x normal volume on an up day")

    dist_high = last.Close / last.High20 - 1
    comps["breakout"] = _ramp(dist_high, -0.08, 0.0)
    if dist_high > -0.01:
        reasons.append("Trading at/near its 20-day high (breakout)")

    # Day traders want movement, but not chaos: 1.5%-5% ATR is ideal.
    if 0.015 <= atr_pct <= 0.05:
        comps["volatility"] = 1.0
    elif atr_pct < 0.015:
        comps["volatility"] = _ramp(atr_pct, 0.005, 0.015)
    else:
        comps["volatility"] = 1.0 - _ramp(atr_pct, 0.05, 0.10)

    score = 100 * sum(STOCK_WEIGHTS[k] * comps[k] for k in STOCK_WEIGHTS)
    price = float(last.Close)
    return StockRating(
        ticker=ticker,
        score=round(score, 1),
        rating=rating_label(score),
        price=price,
        change_pct=float(change * 100),
        rsi=float(last.RSI),
        rel_volume=float(rel_vol),
        atr=float(last.ATR),
        hv20=float(last.HV20),
        # Swing levels: 2 ATR reward vs 1 ATR risk (2:1 reward/risk).
        target=round(price + 2 * float(last.ATR), 2),
        stop=round(price - float(last.ATR), 2),
        components={k: round(v, 3) for k, v in comps.items()},
        reasons=reasons,
    )


def rate_universe(histories: dict[str, pd.DataFrame]) -> list[StockRating]:
    ratings = [rate_stock(t, h) for t, h in histories.items()]
    return sorted((r for r in ratings if r is not None), key=lambda r: r.score, reverse=True)


def ratings_frame(ratings: list[StockRating]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Ticker": r.ticker,
                "Score": r.score,
                "Rating": r.rating,
                "Price": r.price,
                "Chg %": r.change_pct,
                "RSI": r.rsi,
                "Rel Vol": r.rel_volume,
                "Target": r.target,
                "Stop": r.stop,
                "Hist Vol": r.hv20 * 100,
            }
            for r in ratings
        ]
    )
