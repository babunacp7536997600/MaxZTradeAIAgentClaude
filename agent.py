"""
agent.py - the AI Precision Agent's decision gate (pure Python, no Streamlit / network).

The agent does not invent a probability.  It runs every scanned stock through a checklist; only setups that pass EVERY
check are listed as agent picks, and setups that fail exactly one check are listed as 'watch'.  Each pick is shown with the
held-out hit-rate that the Performance Lab measured for the same kind of gate (when the user has run the verification),
so the number next to a pick is a measured one - not the model confidence index.

Checks that were back-tested (they exist in validation.py / precision_lab.py): signal score, strict confluence, regime,
stretch, price-volume footprint.  Checks that are live overlays and NOT back-tested (no historical data): named
bulk / block deal flow and liquidity.  The UI labels each one.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

import validation as V


@dataclass
class Gate:
    horizon: str = "5 Trading Days"
    min_prob: int = 70
    min_abs_score: float = 40.0
    strict: bool = True
    need_regime: bool = True
    need_footprint: bool = True
    max_stretch: float = 2.2
    min_turnover_cr: float = 10.0
    cost_bps: float = 20.0
    use_deal_overlay: bool = True


def levels(res: dict, side: str, gate: Gate, plan: dict | None = None) -> dict:
    """Entry / target / stop.  With a verified `plan` ({'target_atr','stop_atr'}) the levels use exactly the multipliers that were
    validated (sqrt(time) x ATR, same as validation.py); otherwise the dashboard's standard plan for the horizon."""
    t = res["tf"][gate.horizon]
    if not plan:
        return {"entry": t["entry"], "target": t["target"], "stop": t["stop"], "exp_pct": t["exp_pct"], "risk_pct": t["risk_pct"], "source": "dashboard plan"}
    n = V.HORIZONS[gate.horizon][1]
    unit, sign, px = math.sqrt(max(n, 1)) * float(res["atr"]), (1 if side == "BUY" else -1), float(res["price"])
    return {"entry": round(px, 2), "target": round(px + sign * plan["target_atr"] * unit, 2), "stop": round(px - sign * plan["stop_atr"] * unit, 2),
            "exp_pct": round(plan["target_atr"] * unit / px * 100, 2), "risk_pct": round(plan["stop_atr"] * unit / px * 100, 2), "source": "verified plan"}


def evaluate(res: dict, side: str, gate: Gate, foot: dict | None = None, deal_net_cr: float | None = None, plan: dict | None = None) -> list[dict]:
    """Checklist for one stock and one direction -> [{check, ok, detail, tested}]."""
    t = res["tf"][gate.horizon]
    buy = side == "BUY"
    lv = levels(res, side, gate, plan)
    checks = []

    def add(name, ok, detail, tested=True):
        checks.append({"check": name, "ok": bool(ok), "detail": detail, "tested": tested})

    add("Model signal in this direction", t["signal"] == side, f"{t['signal']} (score {t['score']:+.0f})")
    add(f"Signal strength >= {gate.min_abs_score:.0f}", abs(t["score"]) >= gate.min_abs_score and (t["score"] > 0) == buy, f"{t['score']:+.0f}")
    add(f"Confidence index >= {gate.min_prob}%", t["prob"] >= gate.min_prob, f"{t['prob']}%", tested=False)
    if gate.strict:
        add("Strict confluence (EMA stack, MACD, RSI band, CMF, volume)", V.strict_current_confirmation(res, side), "all 6 conditions" if V.strict_current_confirmation(res, side) else "one or more missing")
    if gate.need_regime:
        reg = res.get("regime")
        add("Nifty regime supports the trade", reg is not None and ((reg > 0) if buy else (reg < 0)),
            "n/a" if reg is None else ("above 50-DMA" if reg > 0 else "below 50-DMA" if reg < 0 else "range-bound"))
    ext = res.get("ext", 0.0)
    add(f"Not stretched (<= {gate.max_stretch:g} ATR from EMA21)", (ext <= gate.max_stretch) if buy else (ext >= -gate.max_stretch), f"{ext:+.1f} ATR")
    exp = lv["exp_pct"]
    add(f"Expected move covers costs (>= 5x {gate.cost_bps:.0f} bps)", exp >= 5 * gate.cost_bps / 100, f"{exp:.1f}% expected vs {gate.cost_bps / 100:.2f}% round trip", tested=False)
    to = res.get("turnover_cr")
    add(f"Liquid (avg daily turnover >= ₹{gate.min_turnover_cr:g} Cr)", to is None or to >= gate.min_turnover_cr, "n/a" if to is None else f"₹{to:,.0f} Cr", tested=False)
    if gate.need_footprint:
        if foot and foot.get("ok"):
            want = "Accumulation" if buy else "Distribution"
            flow = foot["flow21_cr"]
            add(f"Price-volume footprint = {want.lower()}", foot["phase"] == want and ((flow > 0) if buy else (flow < 0)),
                f"{foot['phase']}, 21d money-flow proxy ₹{flow:+,.0f} Cr")
            w = foot.get("window")
            if w:
                add("Phase not exhausted (younger than 75% of past phases)", not w["mature"] if foot["phase"] == want else True,
                    f"day {w['age']} vs typical {w['q25']:.0f}-{w['q75']:.0f}")
        else:
            add(f"Price-volume footprint = {'accumulation' if buy else 'distribution'}", False, "no volume footprint available")
    if gate.use_deal_overlay and deal_net_cr is not None:
        add("Disclosed bulk/block deals not against the trade", (deal_net_cr >= 0) if buy else (deal_net_cr <= 0),
            f"net ₹{deal_net_cr:+,.1f} Cr in the window", tested=False)
    return checks


def brief(scan: dict, side: str, gate: Gate, deal_net: dict | None = None, plan: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the whole scan through the gate -> (picks that pass everything, watch-list failing exactly one check)."""
    picks, watch = [], []
    foot_map = scan.get("footprint", {}) or {}
    deal_net = deal_net or {}
    for sym, res in scan["results"].items():
        try:
            ch = evaluate(res, side, gate, foot_map.get(sym), deal_net.get(sym), plan)
        except Exception:
            continue
        failed = [c for c in ch if not c["ok"]]
        if len(failed) > 1:
            continue
        t = res["tf"][gate.horizon]
        lv = levels(res, side, gate, plan)
        fp = foot_map.get(sym) or {}
        row = {"Stock": sym, "Name": res["Name"], "Sector": res.get("sector") or "-", "Side": side, "LTP": round(res["price"], 2),
               "Confidence index": t["prob"], "Score": t["score"], "Status": t["status"], "Entry": lv["entry"], "Target": lv["target"],
               "Stop": lv["stop"], "Exp. move %": lv["exp_pct"], "Risk %": lv["risk_pct"], "Exit rule": t["trail"],
               "Footprint": fp.get("phase", "n/a"), "Checks passed": f"{len(ch) - len(failed)}/{len(ch)}",
               "Failed check": failed[0]["check"] if failed else "-", "_checks": ch}
        (picks if not failed else watch).append(row)
    key = lambda r: (-r["Confidence index"], -abs(r["Score"]))
    return pd.DataFrame(sorted(picks, key=key)), pd.DataFrame(sorted(watch, key=key))


def deal_net_by_symbol(deals: pd.DataFrame) -> dict:
    """Net disclosed Rs Cr (all participants) per stock - the live overlay used by the gate."""
    if deals is None or deals.empty:
        return {}
    return deals.groupby("Symbol")["Signed_Cr"].sum().round(1).to_dict()
