"""
precision_lab.py - universe-wide, out-of-sample verification of the AI Precision Agent's gates.

Why this exists
---------------
A single stock gives 5-20 held-out trades: far too few to say anything about a win rate.  This module pools the same
chronological walk-forward test (`validation.walk_forward_backtest`: next-session Open entry, stop-first on ambiguous
candles, costs deducted) over many symbols, so a result rests on hundreds of independent trades.

How settings are chosen (no peeking)
------------------------------------
Every configuration of (gate, score threshold, target ATR, stop ATR) is run on the whole universe.  The configuration is
SELECTED using the TRAINING-period trades only; the held-out trades of that single configuration are then reported.
The full table (train and test side by side) is returned so over-fitting is visible: a big gap between a configuration's
train and test numbers is the warning sign.

Win rate alone is a trap: a tiny target with a wide stop wins often and still loses money.  Selection therefore always
demands positive expectancy and profit factor > 1 on the training trades.
"""
from __future__ import annotations

import itertools
import math
from typing import Callable

import numpy as np
import pandas as pd

import validation as V

GATES = {"core": dict(strict_filter=False, precision_filter=False),
         "strict": dict(strict_filter=True, precision_filter=False),
         "precision": dict(strict_filter=False, precision_filter=True)}

DEFAULT_GRID = {"gate": ["strict", "precision"], "threshold": [40, 55], "target_atr": [0.6, 0.9, 1.2], "stop_atr": [0.6, 1.0]}


def pooled_metrics(trades: pd.DataFrame | None) -> dict:
    """Trade-level statistics of a pooled set of trades (no compounding across overlapping symbols)."""
    if trades is None or len(trades) == 0:
        return {"trades": 0, "symbols": 0, "wins": 0, "win_rate": None, "ci_low": None, "ci_high": None, "avg_net": None,
                "profit_factor": None, "expectancy_r": None, "target_first_rate": None, "max_dd_r": None,
                "buy_win_rate": None, "sell_win_rate": None, "buy_trades": 0, "sell_trades": 0}
    t = trades.sort_values("exit_date")
    r = t["net_return_pct"].astype(float) / 100.0
    n, wins = len(r), int((r > 0).sum())
    lo, hi = V._wilson_interval(wins, n)
    gp, gl = float(r[r > 0].sum()), float(-r[r < 0].sum())
    pf = gp / gl if gl > 0 else (float("inf") if gp > 0 else None)
    cum_r = t["r_multiple"].astype(float).fillna(0.0).cumsum()
    out = {"trades": n, "symbols": int(t["symbol"].nunique()) if "symbol" in t else 0, "wins": wins, "win_rate": wins / n * 100,
           "ci_low": lo, "ci_high": hi, "avg_net": float(r.mean() * 100), "profit_factor": pf,
           "expectancy_r": float(t["r_multiple"].astype(float).mean()), "target_first_rate": float(t["target_first"].mean() * 100),
           "max_dd_r": float((cum_r - cum_r.cummax()).min())}
    for side, key in (("BUY", "buy"), ("SELL", "sell")):
        s = t[t["side"] == side]
        out[f"{key}_trades"] = int(len(s))
        out[f"{key}_win_rate"] = float((s["net_return_pct"] > 0).mean() * 100) if len(s) else None
    return out


def run_universe(prices: dict, prepared: dict, regime, bench, *, horizon: str, side_mode: str = "BUY + SELL", threshold: int = 40,
                 stop_atr: float = 0.55, target_atr: float = 1.10, cost_bps: float = 20.0, gate: str = "strict",
                 train_fraction: float = 0.65) -> dict:
    """Pool the walk-forward result of every symbol for one configuration."""
    test, train, skipped = [], [], 0
    for sym, df in prices.items():
        try:
            res = V.walk_forward_backtest(df, horizon_label=horizon, side_mode=side_mode, threshold=threshold,
                                          train_fraction=train_fraction, stop_atr=stop_atr, target_atr=target_atr, cost_bps=cost_bps,
                                          regime=regime, bench=bench, prepared=prepared.get(sym), **GATES[gate])
        except ValueError:
            skipped += 1
            continue
        if len(res["trades"]):
            test.append(res["trades"].assign(symbol=sym))
        if len(res["train_trades"]):
            train.append(res["train_trades"].assign(symbol=sym))
    test_df = pd.concat(test, ignore_index=True) if test else pd.DataFrame()
    train_df = pd.concat(train, ignore_index=True) if train else pd.DataFrame()
    return {"test": test_df, "train": train_df, "test_metrics": pooled_metrics(test_df), "train_metrics": pooled_metrics(train_df),
            "skipped": skipped, "config": dict(gate=gate, threshold=threshold, target_atr=target_atr, stop_atr=stop_atr)}


def _eligible(m: dict, min_trades: int) -> bool:
    pf = m.get("profit_factor")
    return bool(m["trades"] >= min_trades and (m["expectancy_r"] or 0) > 0 and pf is not None and pf > 1.0)


def optimize(prices: dict, regime, bench, *, horizon: str, side_mode: str = "BUY + SELL", cost_bps: float = 20.0,
             train_fraction: float = 0.65, grid: dict | None = None, objective: str = "win_rate", min_train_trades: int = 60,
             progress: Callable[[float, str], None] | None = None) -> dict:
    """
    Run the grid, select on TRAIN only, report the selected configuration's held-out result.
    objective: 'win_rate' (highest train win rate among configurations that still have positive expectancy) or 'expectancy'.
    """
    grid = grid or DEFAULT_GRID
    prepared = {}
    for sym, df in prices.items():
        try:
            prepared[sym] = V.prepare(df, regime, bench)
        except Exception:
            continue
    prices = {s: prices[s] for s in prepared}
    combos = list(itertools.product(grid["gate"], grid["threshold"], grid["target_atr"], grid["stop_atr"]))
    rows, runs = [], {}
    for i, (gate, thr, tgt, stp) in enumerate(combos):
        if progress:
            progress(i / max(len(combos), 1), f"Configuration {i + 1}/{len(combos)}: {gate}, score>={thr}, target {tgt}xATR, stop {stp}xATR")
        r = run_universe(prices, prepared, regime, bench, horizon=horizon, side_mode=side_mode, threshold=thr, stop_atr=stp,
                         target_atr=tgt, cost_bps=cost_bps, gate=gate, train_fraction=train_fraction)
        tm, sm = r["train_metrics"], r["test_metrics"]
        key = (gate, thr, tgt, stp)
        runs[key] = r
        rows.append({"Gate": gate, "Min score": thr, "Target ATR": tgt, "Stop ATR": stp,
                     "Train trades": tm["trades"], "Train win %": tm["win_rate"], "Train expectancy R": tm["expectancy_r"],
                     "Train PF": tm["profit_factor"], "Eligible": _eligible(tm, min_train_trades),
                     "Test trades": sm["trades"], "Test win %": sm["win_rate"], "Test CI low %": sm["ci_low"],
                     "Test expectancy R": sm["expectancy_r"], "Test PF": sm["profit_factor"]})
    if progress:
        progress(1.0, "Done")
    table = pd.DataFrame(rows)
    sel_key, sel_reason = None, ""
    el = table[table["Eligible"]]
    if len(el):
        col = "Train win %" if objective == "win_rate" else "Train expectancy R"
        best = el.sort_values(col, ascending=False).iloc[0]
        sel_key = (best["Gate"], int(best["Min score"]), float(best["Target ATR"]), float(best["Stop ATR"]))
        sel_reason = f"Selected on TRAINING trades only ({'highest win rate' if objective == 'win_rate' else 'highest expectancy'} among configurations with positive expectancy and profit factor > 1)."
    else:
        sel_reason = ("No configuration had positive expectancy and profit factor > 1 on the training trades with enough trades. "
                      "That is a valid finding - the agent should not be traded on these settings.")
    selected = runs.get(sel_key) if sel_key else None
    return {"table": table, "selected": selected, "selected_key": sel_key, "reason": sel_reason, "n_symbols": len(prices),
            "horizon": horizon, "cost_bps": cost_bps, "objective": objective}


def goal_check(m: dict, goal: float = 80.0, min_trades: int = 100) -> dict:
    """Transparent pass / fail of the 'verified 80%' claim on HELD-OUT trades."""
    n = m.get("trades", 0)
    wr, lo = m.get("win_rate"), m.get("ci_low")
    pf, ex = m.get("profit_factor"), m.get("expectancy_r")
    checks = [
        (f"At least {min_trades} held-out trades", n >= min_trades, f"{n} trades across {m.get('symbols', 0)} symbols"),
        (f"Held-out win rate >= {goal:.0f}%", wr is not None and wr >= goal, "n/a" if wr is None else f"{wr:.1f}%"),
        (f"95% interval lower bound >= {goal - 10:.0f}%", lo is not None and lo >= goal - 10, "n/a" if lo is None else f"{lo:.1f}%"),
        ("Positive expectancy after costs", ex is not None and ex > 0, "n/a" if ex is None else f"{ex:+.2f} R"),
        ("Profit factor >= 1.3", pf is not None and (pf == float("inf") or pf >= 1.3), "n/a" if pf is None else ("inf" if pf == float("inf") else f"{pf:.2f}")),
    ]
    passed = all(c[1] for c in checks)
    return {"checks": checks, "verified": passed,
            "label": f"VERIFIED >= {goal:.0f}% (held-out)" if passed else f"NOT VERIFIED at {goal:.0f}% - measured {('%.1f%%' % wr) if wr is not None else 'n/a'}"}
