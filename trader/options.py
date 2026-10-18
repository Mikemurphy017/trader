"""Black-Scholes math and the rating algorithm for option contracts.

Only long (bought) calls and puts are rated. Max loss is the premium paid,
and Robinhood's Level 2 options approval is enough to place these trades.

Each contract gets a 0-100 score built from seven weighted components:

    Underlying     25%  the stock's own score (inverted for puts)
    Probability    20%  chance the stock finishes past breakeven at expiry
    Liquidity      20%  bid/ask spread, open interest, volume
    Delta          10%  0.35-0.65 sweet spot (moves with the stock, fair cost);
                        contracts outside 0.20-0.80 are not rated at all
    IV value       10%  implied vol vs. the stock's realised vol (cheap = good)
    Move needed    10%  move to breakeven vs. the market's expected move
    Time           5%   14-45 days to expiry limits time decay
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from .scoring import StockRating, _clip01, _ramp

RISK_FREE_RATE = 0.04

OPTION_WEIGHTS = {
    "underlying": 0.25,
    "probability": 0.20,
    "liquidity": 0.20,
    "delta": 0.10,
    "iv_value": 0.10,
    "move_needed": 0.10,
    "time": 0.05,
}


def _norm_cdf(x: float) -> float:
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def bs_price(spot: float, strike: float, t: float, vol: float, kind: str, r: float = RISK_FREE_RATE) -> float:
    if t <= 0 or vol <= 0:
        intrinsic = spot - strike if kind == "call" else strike - spot
        return max(intrinsic, 0.0)
    d1 = (math.log(spot / strike) + (r + vol**2 / 2) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    if kind == "call":
        return spot * _norm_cdf(d1) - strike * math.exp(-r * t) * _norm_cdf(d2)
    return strike * math.exp(-r * t) * _norm_cdf(-d2) - spot * _norm_cdf(-d1)


def bs_delta(spot: float, strike: float, t: float, vol: float, kind: str, r: float = RISK_FREE_RATE) -> float:
    if t <= 0 or vol <= 0:
        itm = spot > strike if kind == "call" else spot < strike
        return (1.0 if itm else 0.0) * (1 if kind == "call" else -1)
    d1 = (math.log(spot / strike) + (r + vol**2 / 2) * t) / (vol * math.sqrt(t))
    return _norm_cdf(d1) if kind == "call" else _norm_cdf(d1) - 1


def implied_vol(price: float, spot: float, strike: float, t: float, kind: str) -> float | None:
    """Solve for implied volatility by bisection. None if the price is out of bounds."""
    if price <= 0 or t <= 0:
        return None
    lo, hi = 0.01, 5.0
    if not bs_price(spot, strike, t, lo, kind) <= price <= bs_price(spot, strike, t, hi, kind):
        return None
    for _ in range(80):
        mid = (lo + hi) / 2
        if bs_price(spot, strike, t, mid, kind) > price:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def prob_beyond(spot: float, level: float, t: float, vol: float, above: bool, r: float = RISK_FREE_RATE) -> float:
    """Risk-neutral probability the stock finishes above (or below) `level` at expiry."""
    if t <= 0 or vol <= 0 or level <= 0:
        return float(spot > level) if above else float(spot < level)
    d2 = (math.log(spot / level) + (r - vol**2 / 2) * t) / (vol * math.sqrt(t))
    return _norm_cdf(d2) if above else _norm_cdf(-d2)


def round_price(p: float) -> float:
    """Round a limit price to a valid tick: $0.01 under $3, $0.05 at $3 and above."""
    tick = 0.01 if p < 3 else 0.05
    return round(round(p / tick) * tick, 2)


@dataclass
class OptionPick:
    ticker: str
    kind: str  # "call" | "put"
    expiration: date
    strike: float
    dte: int
    bid: float
    ask: float
    mid: float
    limit_price: float
    iv: float
    delta: float
    breakeven: float
    prob_profit: float
    open_interest: int
    volume: int
    spread_pct: float
    score: float
    rating: str
    spot: float
    stock_score: float
    components: dict[str, float] = field(default_factory=dict)
    contract_symbol: str = ""

    @property
    def cost_per_contract(self) -> float:
        return self.limit_price * 100

    def plan(self, account_size: float, risk_pct: float, stop_loss_pct: float = 0.40, take_profit_pct: float = 0.60) -> dict:
        """Position size and exit levels.

        Sizing assumes the stop is honoured: the amount at risk per contract is
        premium * stop_loss_pct, so contracts = risk budget / that amount, but
        never more than half the account is spent on premium.
        """
        risk_budget = account_size * risk_pct
        risk_per_contract = self.cost_per_contract * stop_loss_pct
        by_risk = int(risk_budget // risk_per_contract) if risk_per_contract > 0 else 0
        by_cash = int((account_size * 0.5) // self.cost_per_contract) if self.cost_per_contract > 0 else 0
        contracts = max(0, min(by_risk, by_cash))
        return {
            "contracts": contracts,
            "total_cost": round(contracts * self.cost_per_contract, 2),
            "max_loss": round(contracts * self.cost_per_contract, 2),
            "planned_loss": round(contracts * risk_per_contract, 2),
            "stop_price": round_price(self.limit_price * (1 - stop_loss_pct)),
            "target_price": round_price(self.limit_price * (1 + take_profit_pct)),
            "exit_by_dte": max(1, min(7, self.dte // 3)),
        }


def option_rating_label(score: float) -> str:
    if score >= 75:
        return "A - Solid bet"
    if score >= 62:
        return "B - Good"
    if score >= 50:
        return "C - Speculative"
    return "D - Pass"


def rate_contract(row: pd.Series, kind: str, spot: float, dte: int, stock: StockRating) -> dict | None:
    """Score a single row of an option chain. Returns None if it can't be priced."""
    t = dte / 365
    bid, ask = float(row.get("bid") or 0), float(row.get("ask") or 0)
    strike = float(row["strike"])
    has_quote = bid > 0 and ask > 0 and ask >= bid
    mid = (bid + ask) / 2 if has_quote else float(row.get("lastPrice") or 0)
    if mid <= 0.05 or t <= 0:
        return None

    iv = implied_vol(mid, spot, strike, t, kind) if has_quote else None
    if iv is None:
        iv = float(row.get("impliedVolatility") or 0)
    if not 0.03 < iv < 4:
        return None

    delta = bs_delta(spot, strike, t, iv, kind)
    # Deep ITM ties up lots of capital; far OTM is a lottery ticket. Skip both.
    if not 0.20 <= abs(delta) <= 0.80:
        return None
    breakeven = strike + mid if kind == "call" else strike - mid
    pop = prob_beyond(spot, breakeven, t, iv, above=kind == "call")
    expected_move = spot * iv * math.sqrt(t)
    move_needed = abs(breakeven - spot)
    spread_pct = (ask - bid) / mid if has_quote else 1.0
    oi = int(row.get("openInterest") or 0) if np.isfinite(row.get("openInterest") or 0) else 0
    vol = int(row.get("volume") or 0) if np.isfinite(row.get("volume") or 0) else 0

    abs_delta = abs(delta)
    if 0.35 <= abs_delta <= 0.65:
        delta_pts = 1.0
    elif abs_delta < 0.35:
        delta_pts = _ramp(abs_delta, 0.10, 0.35)
    else:
        delta_pts = 1.0 - _ramp(abs_delta, 0.65, 0.80)

    if 14 <= dte <= 45:
        time_pts = 1.0
    elif dte < 14:
        time_pts = _ramp(dte, 2, 14)
    else:
        time_pts = 1.0 - _ramp(dte, 45, 120) * 0.7

    hv = stock.hv20 if stock.hv20 > 0 else iv
    stock_pts = stock.score / 100 if kind == "call" else 1 - stock.score / 100

    comps = {
        "underlying": stock_pts,
        # A coin-flip (50%) is excellent for a long option; 20% is a lottery ticket.
        "probability": _ramp(pop, 0.15, 0.50),
        "liquidity": (
            0.5 * (1 - _ramp(spread_pct, 0.03, 0.20))
            + 0.3 * _ramp(math.log10(oi + 1), 1.5, 3.5)
            + 0.2 * _ramp(math.log10(vol + 1), 1.0, 3.0)
        ),
        "delta": delta_pts,
        "iv_value": 1 - _ramp(iv / hv, 0.9, 2.0),
        "move_needed": 1 - _ramp(move_needed / expected_move if expected_move else 9, 0.2, 1.2),
        "time": time_pts,
    }
    score = 100 * sum(OPTION_WEIGHTS[k] * _clip01(v) for k, v in comps.items())
    return {
        "strike": strike,
        "bid": bid,
        "ask": ask,
        "mid": mid,
        "iv": iv,
        "delta": delta,
        "breakeven": breakeven,
        "prob_profit": pop,
        "open_interest": oi,
        "volume": vol,
        "spread_pct": spread_pct,
        "score": round(score, 1),
        "components": {k: round(_clip01(v), 3) for k, v in comps.items()},
        "contract_symbol": str(row.get("contractSymbol", "")),
    }


def rate_chain(
    stock: StockRating,
    chains: dict[date, dict[str, pd.DataFrame]],
    today: date,
    kinds: tuple[str, ...] = ("call",),
    min_dte: int = 7,
    max_dte: int = 60,
    min_open_interest: int = 100,
    max_premium: float | None = None,
) -> list[OptionPick]:
    """Rate every contract in `chains` ({expiry: {"calls": df, "puts": df}}).

    `max_premium` (per share) drops contracts too expensive for the account.
    """
    picks: list[OptionPick] = []
    for expiry, sides in chains.items():
        dte = (expiry - today).days
        if not min_dte <= dte <= max_dte:
            continue
        for kind in kinds:
            df = sides.get(kind + "s")
            if df is None or df.empty:
                continue
            # Skip far OTM/ITM strikes: only +/-20% around spot are worth scoring.
            near = df[(df["strike"] >= stock.price * 0.8) & (df["strike"] <= stock.price * 1.2)]
            for _, row in near.iterrows():
                res = rate_contract(row, kind, stock.price, dte, stock)
                if res is None or res["open_interest"] < min_open_interest:
                    continue
                if max_premium is not None and res["mid"] > max_premium:
                    continue
                comps = res.pop("components")
                picks.append(
                    OptionPick(
                        ticker=stock.ticker,
                        kind=kind,
                        expiration=expiry,
                        dte=dte,
                        limit_price=round_price(res["mid"]),
                        rating=option_rating_label(res["score"]),
                        spot=stock.price,
                        stock_score=stock.score,
                        components=comps,
                        **res,
                    )
                )
    return sorted(picks, key=lambda p: p.score, reverse=True)


def picks_frame(picks: list[OptionPick]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Ticker": p.ticker,
                "Type": p.kind.upper(),
                "Strike": p.strike,
                "Expiry": p.expiration.isoformat(),
                "DTE": p.dte,
                "Score": p.score,
                "Rating": p.rating,
                "Limit": p.limit_price,
                "Cost/contract": p.cost_per_contract,
                "Delta": p.delta,
                "IV %": p.iv * 100,
                "Breakeven": p.breakeven,
                "Prob Profit %": p.prob_profit * 100,
                "Spread %": p.spread_pct * 100,
                "Open Int": p.open_interest,
                "Volume": p.volume,
            }
            for p in picks
        ]
    )
