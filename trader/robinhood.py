"""Step-by-step instructions for placing a rated option trade in the Robinhood app.

Robinhood occasionally renames buttons; the flow below matches the mobile app
layout (stock page -> Trade -> Trade options -> chain -> order ticket).
"""

from __future__ import annotations

from .options import OptionPick

ONE_TIME_SETUP = [
    "Enable options: Account (person icon) -> Menu (three lines) -> Investing -> Options trading -> "
    "apply. Buying calls and puts requires **Level 2** approval.",
    "Check your day-trade status: Account -> Menu -> Investing -> Day trade settings. Under the "
    "pattern day trader (PDT) rules a margin account under $25,000 is limited in how many same-day "
    "round trips it can make in 5 trading days; check the current rule there before day trading.",
    "Turn on order notifications (Account -> Menu -> Notifications) so you see fills immediately.",
]


def entry_steps(pick: OptionPick, contracts: int) -> list[str]:
    side = "Call" if pick.kind == "call" else "Put"
    e = pick.expiration
    exp = f"{e:%b} {e.day}, {e.year}"
    qty = max(contracts, 1)
    return [
        f"Open Robinhood and search for **{pick.ticker}**. Tap it to open the stock page.",
        "Tap **Trade** (bottom right), then **Trade options**.",
        f"At the top of the option chain, select **Buy** and **{side}**.",
        f"Swipe the date bar and tap the **{exp}** expiration ({pick.dte} days out).",
        f"Scroll to the **${pick.strike:g} strike** and tap the green **+** next to it, then **Continue**.",
        f"Check the live quote first. The dashboard saw bid ${pick.bid:.2f} / ask ${pick.ask:.2f}. "
        "If the spread has widened a lot or the price moved more than ~10%, re-check the setup before buying.",
        f"On the order ticket choose **Limit order** and set **Limit price = ${pick.limit_price:.2f}** "
        "(the mid-point). Never use a market order on options; the spread can cost you.",
        f"Enter **Contracts = {qty}** (est. cost ${qty * pick.cost_per_contract:,.2f}).",
        "Leave Time in force as **Good for day**, tap **Review**, confirm the breakeven Robinhood "
        f"shows is about ${pick.breakeven:.2f}, then **swipe up to submit**.",
        "If it doesn't fill within a few minutes, cancel and resubmit $0.01-$0.05 higher. Don't chase "
        "more than ~5% above the mid.",
    ]


def exit_steps(pick: OptionPick, plan: dict) -> list[str]:
    return [
        "Once filled, open the position: Home -> scroll to **Options** -> tap the contract.",
        f"**Take profit:** tap **Sell** -> **Limit order** at **${plan['target_price']:.2f}** "
        "-> Time in force **Good till canceled** -> Review -> swipe up. This works the exit for you.",
        f"**Stop loss:** Robinhood doesn't offer bracket (OCO) orders, and the same contracts can't back both a profit order and a stop order at once, so either (a) set a price "
        f"alert at ${plan['stop_price']:.2f} on the option and sell manually if it hits, or (b) instead of "
        f"the profit order, place a **Stop limit** sell with stop ${plan['stop_price']:.2f} and limit "
        f"${max(0.01, plan['stop_price'] - 0.05):.2f}.",
        f"**Time stop:** close the trade by the time it has **{plan['exit_by_dte']} days** left, win or lose. "
        "Time decay speeds up sharply near expiration.",
        "Don't hold to expiration. Robinhood may close in-the-money positions near the end of expiration "
        "day if you lack buying power to exercise, and out-of-the-money contracts expire worthless.",
        "To exit manually: tap the position -> **Sell** -> **Limit order** near the current mid -> swipe up.",
    ]


def instructions_markdown(pick: OptionPick, plan: dict) -> str:
    side = "CALL" if pick.kind == "call" else "PUT"
    head = (
        f"### Buy to open {plan['contracts'] or 1}x {pick.ticker} ${pick.strike:g} {side} "
        f"exp {pick.expiration:%m/%d/%Y} @ ${pick.limit_price:.2f} limit\n"
    )
    parts = [head, "**Before your first trade (one time)**"]
    parts += [f"- {s}" for s in ONE_TIME_SETUP]
    parts.append("\n**Enter the trade**")
    parts += [f"{i}. {s}" for i, s in enumerate(entry_steps(pick, plan["contracts"]), 1)]
    parts.append("\n**Manage and exit**")
    parts += [f"{i}. {s}" for i, s in enumerate(exit_steps(pick, plan), 1)]
    return "\n".join(parts)
