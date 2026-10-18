"""Day Trader Dashboard. Run with:  streamlit run app.py"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from trader import data
from trader.indicators import add_indicators
from trader.options import OPTION_WEIGHTS, picks_frame
from trader.robinhood import instructions_markdown
from trader.scanner import load_histories, scan_options
from trader.scoring import STOCK_WEIGHTS, rate_universe, ratings_frame

st.set_page_config(page_title="Day Trader Dashboard", page_icon="📈", layout="wide")


def md(text: str) -> None:
    """st.markdown treats paired $ signs as LaTeX; escape them so prices render."""
    st.markdown(text.replace("$", "\\$"))

# ----------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Settings")
    source = st.radio("Data source", ["Live (Yahoo Finance)", "Demo (offline sample data)"])
    demo = source.startswith("Demo")
    watchlist = st.text_area("Watchlist (comma separated)", ", ".join(data.DEFAULT_WATCHLIST), height=110)
    tickers = sorted({t.strip().upper() for t in watchlist.replace("\n", ",").split(",") if t.strip()})

    st.subheader("Your account")
    account = st.number_input("Account size ($)", 500.0, 10_000_000.0, 10_000.0, step=500.0)
    risk_pct = st.slider("Risk per trade (% of account)", 0.5, 5.0, 2.0, 0.5) / 100
    stop_pct = st.slider("Option stop loss (% of premium)", 20, 60, 40, 5) / 100
    tp_pct = st.slider("Option profit target (% gain)", 25, 150, 60, 5) / 100

    st.subheader("Options filters")
    dte_range = st.slider("Days to expiration", 1, 120, (7, 45))
    min_oi = st.number_input("Min open interest", 0, 100_000, 250, step=50)
    top_n = st.slider("Scan options on top N stocks", 1, 15, 6)
    include_puts = st.checkbox("Also find puts on bearish stocks", value=False)
    affordable = st.checkbox("Only contracts that fit my risk budget", value=True,
                             help="Hide contracts where even 1 contract would lose more than your risk "
                                  "budget at the stop loss.")
    # Largest per-share premium where 1 contract's loss at the stop fits the risk budget.
    max_premium = account * risk_pct / (100 * stop_pct) if affordable else None
    run = st.button("Refresh scan", type="primary", use_container_width=True)


@st.cache_data(ttl=300, show_spinner=False)
def cached_histories(tickers: tuple[str, ...], demo: bool):
    return load_histories(list(tickers), demo)


@st.cache_data(ttl=300, show_spinner=False)
def cached_options(tickers: tuple[str, ...], demo: bool, top_n, include_puts, dte_range, min_oi, max_premium):
    ratings = rate_universe(cached_histories(tickers, demo))
    return scan_options(
        ratings, demo, top_n=top_n, include_puts=include_puts,
        min_dte=dte_range[0], max_dte=dte_range[1], min_open_interest=min_oi, max_premium=max_premium,
    )


if run:
    st.cache_data.clear()

st.title("📈 Day Trader Dashboard")
st.caption(
    "Educational tool, not financial advice. Scores rank technical setups; they do not guarantee "
    "a stock will rise. Options can lose 100% of the premium. Yahoo data is ~15 min delayed."
)
if demo:
    st.warning("DEMO MODE: prices and option chains are synthetic. Switch to Live in the sidebar for real data.")

if not tickers:
    st.info("Add some tickers to the watchlist.")
    st.stop()

with st.spinner("Scoring stocks..."):
    try:
        histories = cached_histories(tuple(tickers), demo)
    except Exception as exc:
        st.error(f"Couldn't download prices ({exc}). Try Demo mode or check your connection.")
        st.stop()
    ratings = rate_universe(histories)

if not ratings:
    st.error("No usable price history returned. Check the tickers or try Demo mode.")
    st.stop()

missing = sorted(set(tickers) - {r.ticker for r in ratings})
if missing:
    st.caption(f"Skipped (no data or < 60 days history): {', '.join(missing)}")

tab_stocks, tab_opts, tab_how = st.tabs(["🏆 Stock Picks", "🎯 Options Picks", "🧮 How ratings work"])


def color_score(v):
    if v >= 80:
        return "background-color:#1b5e20;color:white"
    if v >= 65:
        return "background-color:#43a047;color:white"
    if v >= 50:
        return "background-color:#fdd835"
    if v >= 40:
        return "background-color:#ffb74d"
    return "background-color:#e53935;color:white"


def color_option(v):
    if v >= 75:
        return "background-color:#1b5e20;color:white"
    if v >= 62:
        return "background-color:#43a047;color:white"
    if v >= 50:
        return "background-color:#fdd835"
    return "background-color:#e53935;color:white"


# ----------------------------------------------------------------- stocks tab
with tab_stocks:
    best = ratings[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Top pick", best.ticker, f"{best.score:.0f}/100")
    c2.metric("Buy-rated stocks", sum(r.score >= 65 for r in ratings), f"of {len(ratings)}")
    c3.metric("Avg score", f"{sum(r.score for r in ratings) / len(ratings):.0f}")
    bear = ratings[-1]
    c4.metric("Weakest", bear.ticker, f"{bear.score:.0f}/100", delta_color="inverse")

    df = ratings_frame(ratings)
    st.dataframe(
        df.style.map(color_score, subset=["Score"]).format(
            {"Score": "{:.0f}", "Price": "${:.2f}", "Chg %": "{:+.2f}%", "RSI": "{:.0f}",
             "Rel Vol": "{:.2f}x", "Target": "${:.2f}", "Stop": "${:.2f}", "Hist Vol": "{:.0f}%"}
        ),
        hide_index=True, use_container_width=True,
    )
    st.caption("Target = price + 2 x ATR, Stop = price - 1 x ATR (a 2:1 reward-to-risk swing plan).")

    pick = st.selectbox("Inspect a stock", [r.ticker for r in ratings])
    r = next(x for x in ratings if x.ticker == pick)
    left, right = st.columns([2, 1])
    with left:
        h = add_indicators(histories[pick]).tail(120)
        fig = go.Figure()
        fig.add_candlestick(x=h.index, open=h.Open, high=h.High, low=h.Low, close=h.Close, name=pick)
        for col, color in [("SMA20", "#42a5f5"), ("SMA50", "#ab47bc"), ("SMA200", "#ff7043")]:
            fig.add_scatter(x=h.index, y=h[col], name=col, line=dict(width=1.3, color=color))
        fig.add_hline(y=r.target, line_dash="dot", line_color="green", annotation_text="target")
        fig.add_hline(y=r.stop, line_dash="dot", line_color="red", annotation_text="stop")
        fig.update_layout(height=430, xaxis_rangeslider_visible=False, margin=dict(l=0, r=0, t=30, b=0),
                          title=f"{pick}: {r.rating} ({r.score:.0f}/100)")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        comp = pd.DataFrame(
            {"Component": list(r.components), "Points": [100 * STOCK_WEIGHTS[k] * v for k, v in r.components.items()],
             "Max": [100 * STOCK_WEIGHTS[k] for k in r.components]}
        )
        bar = go.Figure()
        bar.add_bar(y=comp.Component, x=comp.Max, orientation="h", marker_color="#e0e0e0", name="max")
        bar.add_bar(y=comp.Component, x=comp.Points, orientation="h", marker_color="#43a047", name="earned")
        bar.update_layout(barmode="overlay", height=260, showlegend=False, margin=dict(l=0, r=0, t=30, b=0),
                          title="Score breakdown")
        st.plotly_chart(bar, use_container_width=True)
        st.markdown("**Why:**\n" + "\n".join(f"- {x}" for x in r.reasons) if r.reasons else "_No standout signals._")

# ----------------------------------------------------------------- options tab
with tab_opts:
    with st.spinner("Scanning option chains..."):
        picks, errors = cached_options(tuple(tickers), demo, top_n, include_puts, dte_range, int(min_oi), max_premium)
    for e in errors:
        st.caption(f"⚠️ option chain error, {e}")
    if not picks:
        st.info("No contracts passed the filters. Try lowering min open interest, widening the DTE range, "
                "enabling puts, or adding cheaper stocks to the watchlist"
                + (f" (your risk budget allows premiums up to ${max_premium:.2f}/share)" if max_premium else "")
                + ". Outside market hours Yahoo often shows no bid/ask.")
    else:
        pf = picks_frame(picks)
        st.dataframe(
            pf.style.map(color_option, subset=["Score"]).format(
                {"Strike": "${:g}", "Score": "{:.0f}", "Limit": "${:.2f}", "Cost/contract": "${:,.0f}",
                 "Delta": "{:.2f}", "IV %": "{:.0f}%", "Breakeven": "${:.2f}", "Prob Profit %": "{:.0f}%",
                 "Spread %": "{:.1f}%", "Open Int": "{:,}", "Volume": "{:,}"}
            ),
            hide_index=True, use_container_width=True,
        )
        labels = [f"{p.ticker} ${p.strike:g} {p.kind.upper()} {p.expiration:%m/%d} | score {p.score:.0f}" for p in picks]
        idx = st.selectbox("Pick a contract to get a trade plan + Robinhood steps", range(len(picks)),
                           format_func=lambda i: labels[i])
        p = picks[idx]
        plan = p.plan(account, risk_pct, stop_pct, tp_pct)

        m = st.columns(5)
        m[0].metric("Rating", p.rating.split(" - ")[0], p.rating.split(" - ")[1])
        m[1].metric("Contracts", plan["contracts"])
        m[2].metric("Total cost", f"${plan['total_cost']:,.0f}")
        m[3].metric("Planned loss (at stop)", f"${plan['planned_loss']:,.0f}")
        m[4].metric("Prob. of profit", f"{p.prob_profit:.0%}")
        if plan["contracts"] == 0:
            st.warning(f"One contract (${p.cost_per_contract:,.0f}) risks more than your per-trade budget. "
                       "Consider a cheaper strike, or raise risk % only if you accept that.")

        a, b = st.columns([3, 2])
        with a:
            md(instructions_markdown(p, plan))
        with b:
            st.markdown("**Contract score breakdown**")
            st.dataframe(
                pd.DataFrame({"Component": list(p.components),
                              "Points": [round(100 * OPTION_WEIGHTS[k] * v, 1) for k, v in p.components.items()],
                              "Max": [100 * OPTION_WEIGHTS[k] for k in p.components]}),
                hide_index=True, use_container_width=True,
            )
            md(
                f"- Underlying **{p.ticker}** stock score: **{p.stock_score:.0f}**, spot ${p.spot:.2f}\n"
                f"- Breakeven at expiry: **${p.breakeven:.2f}** "
                f"({(p.breakeven / p.spot - 1) * 100:+.1f}% from here)\n"
                f"- Delta {p.delta:.2f}: option moves ~${abs(p.delta) * 100:.0f} per $1 stock move per contract\n"
                f"- Profit target ${plan['target_price']:.2f} / stop ${plan['stop_price']:.2f}"
            )

# ----------------------------------------------------------------- how-it-works tab
with tab_how:
    st.markdown(
        """
### Stock score (0-100)
| Component | Weight | What earns points |
|---|---|---|
| Trend | 25% | Price above SMA20; SMA20 above SMA50; price above SMA200; SMA50 rising |
| Momentum | 25% | RSI 50-68 (full credit), fading credit when overbought; MACD histogram positive and rising |
| Returns | 15% | 5-day return from -3% to +6% and 20-day return from -5% to +15% scale up linearly |
| Volume | 15% | Relative volume 0.8x to 2x, full credit only on up days (buyers in control) |
| Breakout | 10% | Close within 8% of the 20-day high, full credit at the high |
| Volatility | 10% | ATR of 1.5%-5% of price: enough range to trade, not chaotic |

**Ratings:** 80+ Strong Buy · 65+ Buy · 50+ Watch · 40+ Neutral · below 40 Avoid/Bearish (put candidates).

### Option score (0-100), long calls and puts only
| Component | Weight | What earns points |
|---|---|---|
| Underlying | 25% | The stock score (calls) or 100 minus it (puts) |
| Probability | 20% | Black-Scholes probability the stock finishes past breakeven; 50% = full credit, 15% = zero |
| Liquidity | 20% | Tight bid/ask spread (<3% best), open interest (up to ~3,000), daily volume |
| Delta | 10% | 0.35-0.65 sweet spot; contracts outside 0.20-0.80 delta are skipped entirely |
| IV value | 10% | Implied vol near or below the stock's 20-day realised vol = cheap premium |
| Move needed | 10% | Move to breakeven small relative to the market's expected move (spot x IV x √t) |
| Time | 5% | 14-45 days to expiry: avoids fast theta decay, keeps leverage |

**Ratings:** 75+ A Solid bet · 62+ B Good · 50+ C Speculative · below 50 D Pass.

### Position sizing
Contracts = (account × risk %) ÷ (premium × 100 × stop-loss %), capped so no more than half the account goes
into premium. The plan assumes you honour the stop; the true worst case is the whole premium.

### Limits
Technical scores are backward-looking and don't know about earnings, news, or macro events. **Check the
earnings date before buying options**, since implied volatility collapses after the report. Paper trade a
few weeks before risking real money.
"""
    )
