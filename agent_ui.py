"""agent_ui.py - 🧠 AI Precision Agent page: transparent gate, daily brief and universe-wide verification."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import agent as A
import data as D
import engine as E
import inst_data as ID
import precision_lab as P
import ui
from knowledge import BENCH, UNI

STRETCH = ui.STRETCH
GOAL = 80.0


def gate_from_selection(key, horizon: str, cost_bps: float) -> tuple[A.Gate, dict]:
    """Translate the verified configuration into the live gate + the plan multipliers that were actually tested."""
    gate, thr, tgt, stp = key
    g = A.Gate(horizon=horizon, min_abs_score=float(thr), strict=True, need_regime=(gate == "precision"), need_footprint=True,
               max_stretch=2.2 if gate == "precision" else 3.0, cost_bps=cost_bps)
    return g, {"target_atr": float(tgt), "stop_atr": float(stp)}


def _fmt(x, f="{:.1f}", none="-"):
    return none if x is None or (isinstance(x, float) and pd.isna(x)) else ("inf" if x == float("inf") else f.format(x))


def _goal_block(ver):
    st.markdown("### 🎯 Goal tracker - is the 80% claim actually verified?")
    if not ver or not ver.get("selected"):
        st.warning(f"**NOT VERIFIED.** No verification has been run (or no configuration qualified). Run the universe verification below. "
                   f"The agent only shows a '✅ verified' badge when held-out trades prove ≥ {GOAL:.0f}% with enough trades and positive expectancy.")
        return
    m = ver["selected"]["test_metrics"]
    gc = P.goal_check(m, GOAL)
    (st.success if gc["verified"] else st.error)(gc["label"])
    c = st.columns(5)
    c[0].metric("Held-out win rate", _fmt(m["win_rate"], "{:.1f}%"))
    c[1].metric("95% interval", f"{_fmt(m['ci_low'])}–{_fmt(m['ci_high'])}%")
    c[2].metric("Held-out trades", m["trades"])
    c[3].metric("Expectancy", _fmt(m["expectancy_r"], "{:+.2f} R"))
    c[4].metric("Profit factor", _fmt(m["profit_factor"], "{:.2f}"))
    st.dataframe(pd.DataFrame([{"Criterion": a, "Pass": "✅" if b else "❌", "Measured": c_} for a, b, c_ in gc["checks"]]), hide_index=True, **STRETCH)
    st.caption(f"Verified on {ver['n_symbols']} symbols · horizon {ver['horizon']} · costs {ver['cost_bps']:.0f} bps round trip · next-session Open entries · stop-first on ambiguous candles.")


def _brief(ver):
    st.markdown("### 📋 Agent brief - live setups that pass every check")
    sel_key = ver["selected_key"] if ver and ver.get("selected") else None
    c1, c2, c3 = st.columns([1.2, 1, 1])
    with c1:
        default_h = ver["horizon"] if sel_key and ver["horizon"] in E.TF_LABELS else "5 Trading Days"
        horizon = st.selectbox("Horizon", E.TF_LABELS, index=E.TF_LABELS.index(default_h), key="ag_h")
    with c2:
        side = st.radio("Direction", ["BUY", "SELL"], horizontal=True, key="ag_side")
    with c3:
        use_deals = st.checkbox("Use disclosed-deal overlay", value=True, key="ag_deals",
                                help="Live overlay: blocks a BUY when disclosed bulk/block deals are net selling (and vice versa). Not back-tested.")
    verified_here = bool(sel_key) and ver["horizon"] == horizon
    if verified_here:
        gate, plan = gate_from_selection(sel_key, horizon, ver["cost_bps"])
        st.caption(f"Using the verified configuration: gate **{sel_key[0]}**, score ≥ {sel_key[1]}, target {sel_key[2]}×ATR, stop {sel_key[3]}×ATR (√time scaled).")
    else:
        gate, plan = A.Gate(horizon=horizon), None
        st.caption("No verified configuration for this horizon - using the default gate and the dashboard's standard plan. Run the verification below to attach a measured hit-rate.")
    with st.expander("Adjust gate"):
        a, b, c = st.columns(3)
        gate.min_prob = a.slider("Min confidence index", 50, 90, gate.min_prob, key="ag_prob")
        gate.min_turnover_cr = b.number_input("Min avg daily turnover ₹ Cr", 0.0, 500.0, gate.min_turnover_cr, 5.0, key="ag_liq")
        gate.max_stretch = c.slider("Max stretch from EMA21 (ATR)", 1.0, 4.0, gate.max_stretch, 0.1, key="ag_ext")
        gate.strict = a.checkbox("Strict confluence", gate.strict, key="ag_strict")
        gate.need_regime = b.checkbox("Nifty regime must agree", gate.need_regime, key="ag_reg")
        gate.need_footprint = c.checkbox("Footprint must agree", gate.need_footprint, key="ag_fp")

    with st.spinner("Scanning the universe through the gate…"):
        scan = D.scan_master()
        deal_net = {}
        if use_deals:
            try:
                deals, _ = ID.deals(30)
                deal_net = A.deal_net_by_symbol(deals)
            except Exception:
                deal_net = {}
    if not scan["results"]:
        st.error("Price data could not be loaded (Yahoo Finance unreachable). Press 🔄 Refresh Terminal and retry.")
        return
    picks, watch = A.brief(scan, side, gate, deal_net if use_deals else None, plan)
    if verified_here:
        m = ver["selected"]["test_metrics"]
        side_wr = m.get("buy_win_rate" if side == "BUY" else "sell_win_rate")
        side_n = m.get("buy_trades" if side == "BUY" else "sell_trades")
        if side_wr is not None and side_n:
            st.info(f"Measured held-out hit-rate of this gate for **{side}** trades: **{side_wr:.1f}%** over {side_n} trades "
                    f"(all directions {m['win_rate']:.1f}%, 95% interval {m['ci_low']:.1f}–{m['ci_high']:.1f}%). This is the number to trust, not the confidence index.")
    if picks.empty:
        st.warning(f"No stock passes every check for {side} on {horizon} right now. That is the agent working as intended - waiting is a position. See the near-misses below.")
    else:
        st.success(f"{len(picks)} {side} setup(s) pass every check.")
        show = [c for c in picks.columns if not c.startswith("_")]
        st.dataframe(picks[show].drop(columns=["Failed check"]), hide_index=True, **STRETCH)
        sel = st.selectbox("Why did it pass? Inspect a pick", list(picks["Stock"]), key="ag_inspect")
        row = picks[picks["Stock"] == sel].iloc[0]
        st.dataframe(pd.DataFrame([{"Check": c["check"], "Pass": "✅" if c["ok"] else "❌", "Detail": c["detail"],
                                    "Back-tested": "yes" if c["tested"] else "live overlay only"} for c in row["_checks"]]), hide_index=True, **STRETCH)
        if st.button("🔍 Open full analysis", key="ag_open"):
            ui.reset_dialog_state()
            ui.open_detail("stock", sel, horizon)
    if not watch.empty:
        with st.expander(f"👀 Near-misses ({len(watch)}) - fail exactly one check"):
            st.dataframe(watch[[c for c in watch.columns if not c.startswith("_")]], hide_index=True, **STRETCH)


def _verify():
    st.markdown("### 🔬 Verify the agent on many stocks (out-of-sample)")
    st.caption("Runs the chronological walk-forward test on a whole universe, picks settings using the TRAINING period only, then reports the later held-out trades. "
               "It takes a minute or two; the result is kept for this session.")
    with st.form("agent_verify"):
        c1, c2, c3, c4 = st.columns(4)
        cat = c1.selectbox("Universe", list(UNI), key="agv_cat")
        nmax = c2.slider("Max stocks", 10, 80, 30, 5, key="agv_n")
        horizon = c3.selectbox("Horizon", E.TF_LABELS, index=1, key="agv_h")
        side_mode = c4.selectbox("Direction", ["BUY + SELL", "BUY", "SELL"], key="agv_side")
        c5, c6, c7 = st.columns(3)
        cost = c5.number_input("Round-trip cost + slippage (bps)", 0.0, 300.0, 20.0, 5.0, key="agv_cost")
        objective = c6.radio("Choose settings for", ["Highest win rate (still profitable)", "Highest expectancy"], key="agv_obj")
        tf = c7.slider("History kept for training", 0.50, 0.85, 0.65, 0.05, key="agv_tf")
        go = st.form_submit_button("Run universe verification", type="primary", **STRETCH)
    if go:
        syms = tuple(s + ".NS" for s in UNI[cat].split()[:nmax])
        try:
            with st.spinner("Downloading 5 years of prices…"):
                frames = D.download_daily(syms, "5y")
                bench_df = D.download_daily((BENCH,), "5y").get(BENCH)
            frames = {k: v for k, v in frames.items() if k != BENCH and len(v) >= 400}
            if len(frames) < 5:
                st.error("Fewer than 5 stocks returned enough history (Yahoo may be throttling). Try again in a minute.")
                return
            regime, bench = E.market_regime(bench_df)
            bar = st.progress(0.0, text="Starting…")
            res = P.optimize(frames, regime, bench, horizon=horizon, side_mode=side_mode, cost_bps=cost, train_fraction=tf,
                             objective="win_rate" if objective.startswith("Highest win") else "expectancy",
                             progress=lambda f, t: bar.progress(min(max(f, 0.0), 1.0), text=t))
            bar.empty()
            st.session_state["agent_verified"] = res
            st.rerun()
        except Exception as exc:
            st.error(f"Verification failed: {exc}")

    ver = st.session_state.get("agent_verified")
    if not ver:
        return
    st.markdown(f"**Result** · {ver['n_symbols']} stocks · {ver['horizon']}")
    st.write(ver["reason"])
    if ver.get("selected"):
        sel = ver["selected"]
        tr = sel["test"].sort_values("exit_date")
        if len(tr):
            st.markdown("**Held-out cumulative R of the selected configuration** (1R = the stop distance; trades pooled in exit-date order)")
            st.line_chart(tr.assign(cumR=tr["r_multiple"].astype(float).fillna(0).cumsum()).set_index("exit_date")[["cumR"]], **STRETCH)
            st.download_button("⬇️ Held-out trade ledger (CSV)", tr.to_csv(index=False).encode("utf-8"), "agent_heldout_trades.csv", "text/csv", key="agv_dl")
    t = ver["table"].copy()
    st.markdown("**Every configuration - train vs held-out side by side**")
    st.dataframe(t, hide_index=True, **STRETCH)
    st.caption("Look at the gap between 'Train win %' and 'Test win %'. A configuration whose win rate is high only because the target is tiny and the stop wide shows positive win % "
               "but ~0 or negative expectancy - the selector rejects those. Survivorship bias (today's constituents) and data-source revisions are not removed.")


def page_agent():
    ui.banner("🧠", "AI Precision Agent - fewer trades, every check explained, accuracy measured not promised",
              "Runs the whole market through a multi-check gate (signal, strict confluence, Nifty regime, price-volume footprint, stretch, liquidity, costs, disclosed deals) "
              "and verifies the gate on held-out data.")
    st.info("**About '80-90%'.** No system can guarantee a win rate. A high *measured* held-out win rate is achievable only by trading rarely or with small targets, and that can still lose money. "
            "So this page reports the measured number with its uncertainty, requires positive expectancy, and shows ✅ *verified* only when held-out trades prove it.")
    ver = st.session_state.get("agent_verified")
    _goal_block(ver)
    _brief(ver)
    st.markdown("---")
    _verify()
