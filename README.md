# Day Trader Dashboard

A Streamlit dashboard with three parts:

1. Ranks a watchlist of stocks with a 0-100 **technical score** (trend, momentum, volume, breakout, volatility).
2. Scans option chains on the strongest names (and optionally puts on the weakest) and gives each contract a 0-100 **option score**.
3. Builds a **trade plan** for the contract you pick: position size, limit price, profit target, stop, and time stop. It also writes **step-by-step Robinhood instructions** for entering and exiting the trade.

> **Not financial advice.** The scores rank technical setups from past price data. They don't know about
> earnings, news, or macro events, and they can't guarantee a stock will rise. Long options can lose 100% of the
> premium. Paper trade first.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The dashboard opens at http://localhost:8501. Pick **Live (Yahoo Finance)** in the sidebar for real data
(delayed about 15 minutes), or **Demo** for synthetic offline data. Set your account size and risk per trade,
then:

- **Stock Picks** tab: ranked table, candlestick chart with SMA20/50/200, target and stop lines, and a score breakdown with reasons.
- **Options Picks** tab: rated contracts. Select one to see the trade plan and Robinhood steps.
- **How ratings work** tab: full formulas.

Live option quotes are only reliable during market hours (9:30-4:00 ET). After hours Yahoo often reports
zero bid/ask, so fewer contracts pass the filters.

## The rating algorithms

### Stock score

| Component  | Weight | Signal |
|------------|-------:|--------|
| Trend      | 25% | Price > SMA20, SMA20 > SMA50, price > SMA200, rising SMA50 |
| Momentum   | 25% | RSI 50-68 sweet spot (less credit when overbought), MACD histogram > 0 and rising |
| Returns    | 15% | 5-day and 20-day performance |
| Volume     | 15% | Relative volume vs. the 20-day average, full credit only on up days |
| Breakout   | 10% | Proximity to the 20-day high |
| Volatility | 10% | ATR 1.5%-5% of price (tradeable range) |

80+ Strong Buy · 65+ Buy · 50+ Watch · 40+ Neutral · <40 Avoid/Bearish.
Each stock also gets a swing **target (price + 2×ATR)** and **stop (price − 1×ATR)**.

### Option score (long calls/puts only)

| Component   | Weight | Signal |
|-------------|-------:|--------|
| Underlying  | 25% | Stock score (calls) or 100 − score (puts) |
| Probability | 20% | Black-Scholes probability of finishing past breakeven |
| Liquidity   | 20% | Bid/ask spread %, open interest, volume |
| Delta       | 10% | 0.35-0.65 sweet spot. Contracts outside 0.20-0.80 are skipped |
| IV value    | 10% | Implied vol vs. 20-day realised vol (cheaper is better) |
| Move needed | 10% | Distance to breakeven vs. the expected move (S·IV·√t) |
| Time        | 5%  | 14-45 DTE preferred |

75+ A (solid bet) · 62+ B (good) · 50+ C (speculative) · <50 D (pass).

**Sizing:** contracts = (account × risk%) ÷ (premium × 100 × stop%), capped at 50% of the account in premium.
With **Only contracts that fit my risk budget** on (the default), contracts where even one would break the
budget are hidden.

## Executing a pick in Robinhood

The dashboard generates these steps for each contract, with your exact ticker, strike, expiry, limit price and
quantity filled in. The general flow:

**One-time setup**
- Account → Menu → Investing → **Options trading**: apply for **Level 2** (needed to buy calls and puts).
- Check **Day trade settings** to see the pattern day trader limits that apply to your account.

**Enter**
1. Search the ticker and open the stock page.
2. Tap **Trade** → **Trade options**.
3. Select **Buy** and **Call** (or **Put**) at the top.
4. Choose the expiration date, then tap the green **+** on the strike → **Continue**.
5. Choose **Limit order** and set the limit price to the mid-point shown in the dashboard. Don't use market orders on options.
6. Enter the number of contracts → **Review** → **Swipe up to submit**.
7. If it doesn't fill within a few minutes, raise the limit $0.01-$0.05 and try again. Don't chase.

**Exit**
- Profit target: open the position → **Sell** → Limit at the target price, Good till canceled.
- Stop: Robinhood has no bracket (OCO) orders. Either set a price alert at the stop and sell manually, or use a **Stop limit** sell order instead of the profit order.
- Time stop: close the trade by the "exit by" days-left shown in the plan. Don't hold options into expiration.

Robinhood changes button labels from time to time. If a label differs, the flow is still the same.

## Project layout

```
app.py                 Streamlit dashboard
trader/indicators.py   SMA, RSI, MACD, ATR, historical vol, relative volume
trader/scoring.py      Stock rating algorithm
trader/options.py      Black-Scholes, implied vol, option rating, trade plan
trader/robinhood.py    Robinhood execution instructions
trader/data.py         Yahoo Finance fetchers and offline demo data
trader/scanner.py      Ties it together
tests/                 pytest suite (runs offline)
```

Run the tests with `pytest`.
