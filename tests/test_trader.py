from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from trader import data
from trader.indicators import rsi
from trader.options import bs_delta, bs_price, implied_vol, prob_beyond, rate_chain, round_price
from trader.robinhood import instructions_markdown
from trader.scanner import scan
from trader.scoring import rate_stock


def trending(days=260, drift=0.003, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(drift, 0.01, days)))
    idx = pd.bdate_range(end="2026-10-16", periods=days)
    vol = rng.integers(1_000_000, 2_000_000, days)
    return pd.DataFrame(
        {"Open": close * 0.998, "High": close * 1.01, "Low": close * 0.99, "Close": close, "Volume": vol},
        index=idx,
    )


def test_rsi_bounds():
    r = rsi(trending()["Close"]).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_uptrend_outscores_downtrend():
    up = rate_stock("UP", trending(drift=0.004))
    down = rate_stock("DN", trending(drift=-0.004))
    assert up.score > 65 > 40 > down.score
    assert up.bias == "bullish" and down.bias == "bearish"
    assert up.stop < up.price < up.target


def test_short_history_is_skipped():
    assert rate_stock("X", trending(days=30)) is None


def test_black_scholes_put_call_parity():
    s, k, t, v, r = 100, 105, 0.25, 0.3, 0.04
    c, p = bs_price(s, k, t, v, "call"), bs_price(s, k, t, v, "put")
    assert c - p == pytest.approx(s - k * np.exp(-r * t), abs=1e-6)
    assert 0 < bs_delta(s, k, t, v, "call") < 1
    assert -1 < bs_delta(s, k, t, v, "put") < 0


def test_implied_vol_round_trip():
    price = bs_price(50, 52, 30 / 365, 0.42, "call")
    assert implied_vol(price, 50, 52, 30 / 365, "call") == pytest.approx(0.42, abs=1e-4)


def test_prob_beyond_symmetry():
    assert prob_beyond(100, 100, 0.1, 0.3, True) + prob_beyond(100, 100, 0.1, 0.3, False) == pytest.approx(1)


def test_round_price_ticks():
    assert round_price(1.234) == 1.23
    assert round_price(4.12) == 4.10


def test_rate_chain_and_plan():
    stock = rate_stock("UP", trending(drift=0.004))
    today = date(2026, 10, 18)
    chains = data.demo_option_chains("UP", stock.price, stock.hv20, today)
    picks = rate_chain(stock, chains, today, ("call",), 7, 60, 0)
    assert picks and all(p.kind == "call" for p in picks)
    assert picks == sorted(picks, key=lambda p: p.score, reverse=True)
    best = picks[0]
    assert 0 <= best.score <= 100 and best.breakeven > best.strike
    plan = best.plan(10_000, 0.02, 0.4, 0.6)
    assert plan["planned_loss"] <= 200 + 1e-9
    assert plan["stop_price"] < best.limit_price < plan["target_price"]
    md = instructions_markdown(best, plan)
    assert "Trade options" in md and f"${best.limit_price:.2f}" in md and "Limit" in md


def test_puts_on_bearish_names():
    stock = rate_stock("DN", trending(drift=-0.004))
    today = date(2026, 10, 18)
    chains = data.demo_option_chains("DN", stock.price, stock.hv20, today)
    picks = rate_chain(stock, chains, today, ("put",), 7, 60, 0)
    assert picks and all(p.breakeven < p.strike for p in picks)


def test_demo_scan_end_to_end():
    ratings, picks, errors = scan(data.DEFAULT_WATCHLIST, demo=True, include_puts=True)
    assert len(ratings) == len(data.DEFAULT_WATCHLIST)
    assert not errors
    assert all(7 <= p.dte <= 60 for p in picks)


def test_delta_band_and_max_premium():
    stock = rate_stock("UP", trending(drift=0.004))
    today = date(2026, 10, 18)
    chains = data.demo_option_chains("UP", stock.price, stock.hv20, today)
    picks = rate_chain(stock, chains, today, ("call",), 7, 60, 0, max_premium=2.0)
    assert picks
    assert all(0.20 <= abs(p.delta) <= 0.80 and p.mid <= 2.0 for p in picks)
