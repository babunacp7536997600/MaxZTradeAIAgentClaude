"""Strict, chronological out-of-sample trade validation for Trader Dashboard Pro.

This module deliberately keeps backtest assumptions explicit:
- signal is generated after a daily candle closes;
- entry is next session's open (never the signal candle's close);
- ATR and all indicators are taken from the signal candle only;
- stop/target checks follow the subsequent OHLC sequence;
- if both stop and target are inside the same daily candle, stop is assumed first;
- round-trip costs are deducted from every trade;
- only fully observable test-window trades are included.

The strategy rules are frozen, not tuned on the test segment. This is a chronological
out-of-sample test; it is not a guarantee of future performance.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

import engine as E

PROBABILITY_BINS = [(30.0, 45.0), (45.0, 60.0), (60.0, 75.0), (75.0, 101.0)]
HORIZONS = {
    "1 Day": ("intraday", 1),
    "2 Trading Days": ("short", 2),
    "3 Trading Days": ("short", 3),
    "4 Trading Days": ("short", 4),
    "5 Trading Days": ("short", 5),
    "2 Weeks": ("short", 10),
    "2-5 Weeks": ("swing", 15),
    "1 Month": ("swing", 21),
    "3 Months": ("positional", 63),
    "6 Months+": ("long", 126),
}


def strict_current_confirmation(result: dict[str, Any], side: str) -> bool:
    """Apply the Performance Lab's strict confluence gate to the latest scan row."""
    if side not in ("BUY", "SELL") or not bool(result.get("has_volume", False)):
        return False
    levels = result.get("levels") or {}
    price = result.get("price")
    ema9, ema21 = levels.get("ema9"), levels.get("ema21")
    rsi, macd_hist, cmf = result.get("rsi"), result.get("macd_hist"), result.get("cmf")
    vol_spike = result.get("vol_spike", 0.0)
    vals = (price, ema9, ema21, rsi, macd_hist, cmf, vol_spike)
    if not all(v is not None and np.isfinite(float(v)) for v in vals):
        return False
    if side == "BUY":
        return bool(price > ema21 and ema9 > ema21 and macd_hist > 0 and 50 <= rsi <= 72 and cmf > 0 and vol_spike >= 0.9)
    return bool(price < ema21 and ema9 < ema21 and macd_hist < 0 and 28 <= rsi <= 50 and cmf < 0 and vol_spike >= 0.9)


def _wilson_interval(wins: int, n: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if n <= 0:
        return None, None
    p = wins / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt((p * (1 - p) / n) + z * z / (4 * n * n)) / den
    return max(0.0, (centre - half) * 100), min(100.0, (centre + half) * 100)


def _strict_mask(d: pd.DataFrame, side: str) -> pd.Series:
    """Higher-confluence confirmation filter; intentionally selective."""
    vol_ok = d.get("HAS_VOL", pd.Series(False, index=d.index)).fillna(False).astype(bool)
    if side == "BUY":
        return (
            (d["Close"] > d["EMA21"])
            & (d["EMA9"] > d["EMA21"])
            & (d["MACD_HIST"] > 0)
            & d["RSI"].between(50, 72)
            & (d["CMF20"] > 0)
            & (vol_ok & (d["VOL_SPIKE"] >= 0.9))
        ).fillna(False)
    return (
        (d["Close"] < d["EMA21"])
        & (d["EMA9"] < d["EMA21"])
        & (d["MACD_HIST"] < 0)
        & d["RSI"].between(28, 50)
        & (d["CMF20"] < 0)
        & (vol_ok & (d["VOL_SPIKE"] >= 0.9))
    ).fillna(False)


def _precision_mask(d: pd.DataFrame, side: str, reg: pd.Series | None = None) -> pd.Series:
    """
    Strictest gate: strict confluence + not stretched + rising/falling SMA50 + real money-flow footprint + Nifty regime aligned.
    Deliberately selective; the Performance Lab / universe test reports whether it actually helps.
    """
    base = _strict_mask(d, side)
    ext, obv, cmf = d["EXT"], d["OBV_SLOPE"], d["CMF20"]
    slope = d["S50"] - d["S50"].shift(5)
    if side == "BUY":
        m = base & ext.between(-0.5, 2.2) & (obv > 0) & (cmf > 0.05) & (slope > 0) & d["RSI"].between(52, 68)
        if reg is not None:
            m &= reg > 0
    else:
        m = base & ext.between(-2.2, 0.5) & (obv < 0) & (cmf < -0.05) & (slope < 0) & d["RSI"].between(32, 48)
        if reg is not None:
            m &= reg < 0
    return m.fillna(False)


def prepare(prices: pd.DataFrame, regime: pd.Series | None = None, bench: pd.DataFrame | None = None):
    """Indicators + family scores computed once, so a parameter grid can reuse them for every configuration."""
    d = E.ind(prices)
    flags = E.pattern_flags(d)
    return d, E.family_scores(d, flags, regime, bench)


def _simulate_trade(
    d: pd.DataFrame,
    signal_i: int,
    side: str,
    horizon: int,
    atr: float,
    stop_atr: float,
    target_atr: float,
    cost_bps: float,
) -> dict[str, Any] | None:
    """Simulate one trade from next-session open; return None if not fully observable."""
    entry_i = signal_i + 1
    last_i = entry_i + horizon - 1
    if entry_i >= len(d) or last_i >= len(d):
        return None

    sign = 1 if side == "BUY" else -1
    raw_entry = float(d["Open"].iloc[entry_i])
    if not np.isfinite(raw_entry) or raw_entry <= 0 or not np.isfinite(atr) or atr <= 0:
        return None

    # Keep the original project's sqrt(time) ATR scaling, but allow independent
    # stop/target multipliers in the Performance Lab.
    unit = math.sqrt(max(horizon, 1)) * atr
    stop_dist = stop_atr * unit
    target_dist = target_atr * unit
    entry = raw_entry
    stop = entry - sign * stop_dist
    target = entry + sign * target_dist
    exit_price = float(d["Close"].iloc[last_i])
    exit_i = last_i
    reason = "TIME EXIT"
    target_first = False
    stop_first = False

    for j in range(entry_i, last_i + 1):
        op = float(d["Open"].iloc[j])
        hi = float(d["High"].iloc[j])
        lo = float(d["Low"].iloc[j])
        if not all(np.isfinite(x) for x in (op, hi, lo)):
            continue

        # Opening gaps are ordered before the intraday high/low range.
        if side == "BUY":
            if op <= stop:
                exit_price, exit_i, reason, stop_first = op, j, "GAP THROUGH STOP", True
                break
            if op >= target:
                exit_price, exit_i, reason, target_first = op, j, "GAP THROUGH TARGET", True
                break
            stop_hit, target_hit = lo <= stop, hi >= target
        else:
            if op >= stop:
                exit_price, exit_i, reason, stop_first = op, j, "GAP THROUGH STOP", True
                break
            if op <= target:
                exit_price, exit_i, reason, target_first = op, j, "GAP THROUGH TARGET", True
                break
            stop_hit, target_hit = hi >= stop, lo <= target

        if stop_hit and target_hit:
            # OHLC daily data cannot identify the path. Prefer the adverse outcome.
            exit_price, exit_i, reason, stop_first = stop, j, "BOTH TOUCHED (STOP-FIRST)", True
            break
        if stop_hit:
            exit_price, exit_i, reason, stop_first = stop, j, "STOP LOSS", True
            break
        if target_hit:
            exit_price, exit_i, reason, target_first = target, j, "TARGET", True
            break

    gross = sign * (exit_price / entry - 1.0)
    cost = max(float(cost_bps), 0.0) / 10_000.0
    net = gross - cost
    risk_fraction = stop_dist / entry
    return {
        "signal_date": pd.Timestamp(d.index[signal_i]),
        "entry_date": pd.Timestamp(d.index[entry_i]),
        "exit_date": pd.Timestamp(d.index[exit_i]),
        "side": side,
        "entry": entry,
        "target": target,
        "stop": stop,
        "exit": exit_price,
        "gross_return_pct": gross * 100,
        "cost_pct": cost * 100,
        "net_return_pct": net * 100,
        "r_multiple": net / risk_fraction if risk_fraction > 0 else np.nan,
        "hold_bars": int(exit_i - entry_i + 1),
        "exit_reason": reason,
        "target_first": bool(target_first),
        "stop_first": bool(stop_first),
    }


def _calibrated_probability(train_trades: list[dict[str, Any]], side: str, abs_score: float) -> tuple[float | None, int]:
    """Estimate P(net-positive trade) from training-only outcomes, shrunk to side prior."""
    same_side = [t for t in train_trades if t["side"] == side]
    n_all = len(same_side)
    if n_all < 8:
        return None, n_all
    wins_all = sum(1 for t in same_side if t["net_return_pct"] > 0)
    prior = wins_all / n_all
    # The bucket lookup is based on the score band containing this test signal.
    lo, hi = next(((lo, hi) for lo, hi in PROBABILITY_BINS if lo <= abs_score < hi), (30.0, 101.0))
    bucket = [t for t in same_side if lo <= abs(float(t["score"])) < hi]
    n_bin = len(bucket)
    wins_bin = sum(1 for t in bucket if t["net_return_pct"] > 0)
    shrink = 15.0
    p = (wins_bin + shrink * prior) / (n_bin + shrink)
    return float(np.clip(p * 100, 1, 99)), n_bin


def _metrics(trades: pd.DataFrame) -> dict[str, Any]:
    if trades is None or trades.empty:
        return {
            "trades": 0, "wins": 0, "win_rate": None, "ci_low": None, "ci_high": None,
            "target_first_rate": None, "avg_return": None, "median_return": None,
            "avg_win": None, "avg_loss": None, "profit_factor": None,
            "expectancy_r": None, "total_return": None, "max_drawdown": None,
            "mean_predicted_win_probability": None, "brier_score": None, "calibrated_trades": 0,
            "equity_curve": pd.DataFrame(),
        }
    r = trades["net_return_pct"].astype(float) / 100.0
    wins = int((r > 0).sum())
    n = len(r)
    lo, hi = _wilson_interval(wins, n)
    eq = (1 + r).cumprod()
    peak = eq.cummax()
    dd = (eq / peak - 1) * 100
    pos = r[r > 0]
    neg = r[r < 0]
    gross_profit = float(pos.sum())
    gross_loss = float(-neg.sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else (float("inf") if gross_profit > 0 else None)
    target_rate = float(trades["target_first"].mean() * 100)
    calibrated = trades[trades["predicted_win_probability"].notna()] if "predicted_win_probability" in trades else pd.DataFrame()
    brier = None
    mean_pred = None
    if len(calibrated):
        p_hat = calibrated["predicted_win_probability"].astype(float) / 100.0
        y = (calibrated["net_return_pct"].astype(float) > 0).astype(float)
        brier = float(((p_hat - y) ** 2).mean())
        mean_pred = float(p_hat.mean() * 100)
    return {
        "trades": n,
        "wins": wins,
        "win_rate": wins / n * 100,
        "ci_low": lo,
        "ci_high": hi,
        "target_first_rate": target_rate,
        "avg_return": float(r.mean() * 100),
        "median_return": float(r.median() * 100),
        "avg_win": float(pos.mean() * 100) if len(pos) else None,
        "avg_loss": float(neg.mean() * 100) if len(neg) else None,
        "profit_factor": pf,
        "expectancy_r": float(trades["r_multiple"].mean()),
        "total_return": float((eq.iloc[-1] - 1) * 100),
        "max_drawdown": float(dd.min()),
        "mean_predicted_win_probability": mean_pred,
        "brier_score": brier,
        "calibrated_trades": int(len(calibrated)),
        "equity_curve": pd.DataFrame({"Trade": np.arange(1, n + 1), "Equity (start=1)": eq.values,
                                       "Drawdown %": dd.values}, index=trades["exit_date"]),
    }


def walk_forward_backtest(
    prices: pd.DataFrame,
    horizon_label: str = "5 Trading Days",
    side_mode: str = "BUY + SELL",
    threshold: int = 40,
    train_fraction: float = 0.65,
    stop_atr: float = 0.55,
    target_atr: float = 1.10,
    cost_bps: float = 20.0,
    strict_filter: bool = False,
    regime: pd.Series | None = None,
    bench: pd.DataFrame | None = None,
    precision_filter: bool = False,
    prepared: tuple | None = None,
) -> dict[str, Any]:
    """Run a chronological holdout test and return trades, metrics and metadata.

    The rules/threshold are not fitted to the holdout. The first `train_fraction`
    of observations are reserved as warm-up/training history; only later signals
    are traded and scored. A signal executes at the next session's Open.
    """
    if horizon_label not in HORIZONS:
        raise ValueError(f"Unsupported horizon: {horizon_label}")
    if prices is None or len(prices) < 260:
        raise ValueError("At least 260 daily OHLCV rows are required for a meaningful test.")
    if not 0.50 <= float(train_fraction) <= 0.85:
        raise ValueError("Training split must be between 50% and 85%.")
    if not 20 <= int(threshold) <= 90:
        raise ValueError("Score threshold must be between 20 and 90.")
    if stop_atr <= 0 or target_atr <= 0:
        raise ValueError("Stop and target ATR multipliers must be positive.")

    d, scores = prepared if prepared is not None else prepare(prices, regime, bench)
    family, horizon = HORIZONS[horizon_label]
    score = scores[family].astype(float)
    n_rows = len(d)
    split_i = max(220, int(n_rows * float(train_fraction)))
    # A full horizon must fit inside the test set; otherwise the outcome is censored.
    last_signal_i = n_rows - horizon - 1
    if split_i >= last_signal_i:
        raise ValueError("Not enough post-split candles for this horizon. Use a shorter horizon or a longer history.")

    reg_al = regime.reindex(d.index, method="ffill").fillna(0.0) if regime is not None and len(regime) else None
    if precision_filter:
        strict_buy, strict_sell = _precision_mask(d, "BUY", reg_al), _precision_mask(d, "SELL", reg_al)
    elif strict_filter:
        strict_buy, strict_sell = _strict_mask(d, "BUY"), _strict_mask(d, "SELL")
    else:
        strict_buy = strict_sell = pd.Series(True, index=d.index)

    def _candidate_directions(i: int, s: float) -> list[str]:
        dirs = []
        if side_mode in ("BUY", "BUY + SELL") and s >= threshold and bool(strict_buy.iloc[i]):
            dirs.append("BUY")
        if side_mode in ("SELL", "BUY + SELL") and s <= -threshold and bool(strict_sell.iloc[i]):
            dirs.append("SELL")
        return dirs

    # Build training outcomes entirely before the held-out split. The calibration
    # maps score bands to net-positive trade rates; no test outcome is used.
    train_trades: list[dict[str, Any]] = []
    train_next_available_i = 220
    train_last_signal_i = min(split_i - horizon - 1, n_rows - horizon - 1)
    for i in range(220, train_last_signal_i + 1):
        s_train = score.iloc[i]
        if not np.isfinite(s_train) or i < train_next_available_i:
            continue
        for direction in _candidate_directions(i, float(s_train)):
            tr_train = _simulate_trade(d, i, direction, horizon, float(d["ATR"].iloc[i]),
                                       float(stop_atr), float(target_atr), float(cost_bps))
            if tr_train is not None and int(d.index.get_indexer([tr_train["exit_date"]])[0]) < split_i:
                tr_train["score"] = float(s_train)
                train_trades.append(tr_train)
                train_next_available_i = int(d.index.get_indexer([tr_train["exit_date"]])[0]) + 1
            break

    train_counts = {
        "BUY": sum(1 for t in train_trades if t["side"] == "BUY"),
        "SELL": sum(1 for t in train_trades if t["side"] == "SELL"),
    }

    trades: list[dict[str, Any]] = []
    next_available_i = split_i
    candidates = []
    for i in range(split_i, last_signal_i + 1):
        s = score.iloc[i]
        if not np.isfinite(s):
            continue
        for direction in _candidate_directions(i, float(s)):
            candidates.append((i, direction, float(s)))

    for i, direction, s in candidates:
        if i < next_available_i:
            continue
        atr = float(d["ATR"].iloc[i])
        tr = _simulate_trade(d, i, direction, horizon, atr, float(stop_atr), float(target_atr), float(cost_bps))
        if tr is None:
            continue
        tr["score"] = s
        tr["signal_strength"] = round(abs(s), 1)
        predicted, calibration_n = _calibrated_probability(train_trades, direction, abs(s))
        tr["predicted_win_probability"] = predicted
        tr["calibration_samples"] = int(calibration_n)
        trades.append(tr)
        # Only one open position per symbol at a time; avoid overlapping labels.
        next_available_i = int(d.index.get_indexer([tr["exit_date"]])[0]) + 1

    train_df = pd.DataFrame(train_trades)
    trade_df = pd.DataFrame(trades)
    if len(trade_df):
        trade_df = trade_df.sort_values(["entry_date", "signal_date"]).reset_index(drop=True)
    metrics = _metrics(trade_df)
    return {
        "trades": trade_df,
        "metrics": metrics,
        "train_trades": train_df,
        "train_metrics": _metrics(train_df),
        "metadata": {
            "horizon": horizon_label,
            "holding_bars": horizon,
            "family": family,
            "split_date": pd.Timestamp(d.index[split_i]),
            "test_start": pd.Timestamp(d.index[split_i]),
            "test_end": pd.Timestamp(d.index[-1]),
            "bars": n_rows,
            "train_bars": split_i,
            "test_bars": n_rows - split_i,
            "train_trade_counts": train_counts,
            "train_calibration_trades": len(train_trades),
            "test_candidate_signals": len(candidates),
            "threshold": int(threshold),
            "side_mode": side_mode,
            "strict_filter": bool(strict_filter),
            "precision_filter": bool(precision_filter),
            "stop_atr": float(stop_atr),
            "target_atr": float(target_atr),
            "cost_bps_round_trip": float(cost_bps),
            "entry_rule": "Next trading session Open",
            "same_bar_rule": "Stop assumed first when both target and stop are touched",
        },
    }
