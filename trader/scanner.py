"""Glue: fetch data, rate stocks, then rate options on the strongest/weakest names."""

from __future__ import annotations

from datetime import date

from . import data
from .options import OptionPick, rate_chain
from .scoring import StockRating, rate_universe


def load_histories(tickers: list[str], demo: bool) -> dict:
    if demo:
        return {t: data.demo_history(t) for t in tickers}
    return data.fetch_history(tickers)


def scan_options(
    ratings: list[StockRating],
    demo: bool,
    top_n: int = 5,
    include_puts: bool = False,
    min_dte: int = 7,
    max_dte: int = 60,
    min_open_interest: int = 100,
    per_ticker: int = 3,
    max_premium: float | None = None,
) -> tuple[list[OptionPick], list[str]]:
    """Rate calls on the top-N bullish names (and puts on bearish names if asked).

    Returns (picks, errors). Only the best `per_ticker` contracts per name and
    side are kept so one ticker can't flood the list.
    """
    today = date.today()
    targets: list[tuple[StockRating, tuple[str, ...]]] = [
        (r, ("call",)) for r in ratings if r.bias == "bullish"
    ][:top_n]
    if include_puts:
        bearish = [r for r in reversed(ratings) if r.bias == "bearish"][:top_n]
        targets += [(r, ("put",)) for r in bearish]

    picks: list[OptionPick] = []
    errors: list[str] = []
    for stock, kinds in targets:
        try:
            chains = (
                data.demo_option_chains(stock.ticker, stock.price, stock.hv20, today)
                if demo
                else data.fetch_option_chains(stock.ticker, max_dte=max_dte)
            )
        except Exception as exc:  # network / Yahoo hiccups shouldn't kill the scan
            errors.append(f"{stock.ticker}: {exc}")
            continue
        rated = rate_chain(stock, chains, today, kinds, min_dte, max_dte, min_open_interest, max_premium)
        for kind in kinds:
            picks += [p for p in rated if p.kind == kind][:per_ticker]
    return sorted(picks, key=lambda p: p.score, reverse=True), errors


def scan(tickers: list[str], demo: bool, **option_kwargs) -> tuple[list[StockRating], list[OptionPick], list[str]]:
    ratings = rate_universe(load_histories(tickers, demo))
    picks, errors = scan_options(ratings, demo, **option_kwargs)
    return ratings, picks, errors
