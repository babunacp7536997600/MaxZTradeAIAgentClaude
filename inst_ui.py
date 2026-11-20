"""inst_ui.py - 🏛️ Institutional Holdings Radar page (who buys / sells what, how much, for how long)."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import data as D
import engine as E
import inst_data as ID
import inst_engine as IE
import ui
from knowledge import SECTOR_STOCKS, STOCK_NAMES, sector_of

STRETCH = ui.STRETCH
WINDOWS = {"1 week": 7, "1 month": 30, "3 months": 90, "6 months": 180}


# ------------------------------------------------------------------------------
# Uploads (processed first so every tab sees them)
# ------------------------------------------------------------------------------
def _uploads():
    with st.expander("📤 Add your own disclosures (recommended - these are the real, official numbers)", expanded=False):
        st.caption("Free public sources do not give fund-house or quarterly category holdings through a stable API, so the dependable route is to "
                   "upload the official files. Everything stays in this browser session only.")
        a, b, c = st.columns(3)
        with a:
            st.markdown("**1. NSE bulk / block deals CSV**")
            st.caption("From NSE → Reports → Bulk deals / Block deals (columns: Date, Symbol, Client Name, Buy/Sell, Quantity Traded, Trade Price).")
            fs = st.file_uploader("Deals files", type=["csv", "xlsx"], accept_multiple_files=True, key="inst_deals_up", label_visibility="collapsed")
            if fs:
                frames = []
                for f in fs:
                    try:
                        frames.append(IE.normalise_deals(ID.csv_with_header(f)))
                    except Exception:
                        st.warning(f"Could not read {f.name}")
                good = [x for x in frames if len(x)]
                st.session_state["inst_upload_deals"] = pd.concat(good, ignore_index=True) if good else None
                st.success(f"{sum(len(x) for x in good):,} deal rows loaded") if good else st.warning("No valid deal rows found (check column names).")
        with b:
            st.markdown("**2. AMC monthly portfolio files**")
            st.caption("Each fund house publishes its monthly portfolio (Excel). Upload one or more months; the app finds the equity lines. Edit fund house / month if guessed wrong.")
            fs = st.file_uploader("AMC portfolios", type=["xlsx", "xls", "csv"], accept_multiple_files=True, key="inst_amc_up", label_visibility="collapsed")
            if fs:
                meta = pd.DataFrame({"File": [f.name for f in fs], "Fund house": [ID.guess_fund_house(f.name) for f in fs],
                                     "Month": [ID.guess_month(f.name).strftime("%Y-%m") for f in fs]})
                meta = st.data_editor(meta, hide_index=True, disabled=["File"], key="inst_amc_meta", **STRETCH)
                frames = []
                for f, (_, row) in zip(fs, meta.iterrows()):
                    try:
                        for sheet, raw in ID.read_any(f).items():
                            part = IE.parse_amc_portfolio(raw, str(row["Fund house"]), pd.Timestamp(str(row["Month"]) + "-01"), "" if sheet == "csv" else str(sheet))
                            if len(part):
                                frames.append(part)
                    except Exception:
                        st.warning(f"Could not read {f.name}")
                st.session_state["inst_upload_mf"] = pd.concat(frames, ignore_index=True) if frames else None
                st.success(f"{sum(len(x) for x in frames):,} equity lines from {len(fs)} file(s)") if frames else st.warning("No equity lines found - is it a portfolio-disclosure sheet?")
        with c:
            st.markdown("**3. Shareholding pattern (per stock)**")
            st.caption("CSV/Excel with quarters and categories (Promoter, FII, DII, Mutual Funds, Public). Rows or columns can be the quarters.")
            sym = st.text_input("Stock symbol", key="inst_sh_sym", placeholder="e.g. RELIANCE").strip().upper()
            f = st.file_uploader("Shareholding file", type=["csv", "xlsx"], key="inst_sh_up", label_visibility="collapsed")
            if f is not None and sym:
                try:
                    raw = ID.csv_with_header(f)
                    long = IE.parse_shareholding(raw)
                    if len(long):
                        st.session_state.setdefault("inst_upload_sh", {})[sym] = long
                        st.success(f"{sym}: {long['Quarter'].nunique()} quarters, {long['Category'].nunique()} categories loaded")
                    else:
                        st.warning("Could not recognise quarters / categories in this file.")
                except Exception:
                    st.warning("Could not read the file.")
            elif f is not None:
                st.info("Enter the stock symbol this file belongs to.")


def _cat_metrics(deals):
    tot = deals.groupby("Category")["Signed_Cr"].sum() if len(deals) else pd.Series(dtype=float)
    cols = st.columns(5)
    for col, cat in zip(cols, IE.CATEGORIES):
        v = float(tot.get(cat, 0.0))
        col.metric(f"{cat} net", f"₹{v:,.0f} Cr", "buying 🟢" if v > 0 else "selling 🔴" if v < 0 else "-")


# ------------------------------------------------------------------------------
def page_institutions():
    ui.banner("🏛️", "Institutional Holdings Radar - who is buying / selling, how much, in which sector & stock, for how long",
              "Named FII / FPI, mutual-fund houses, insurers, DIIs and other big players from disclosed bulk/block deals, uploaded AMC & shareholding files, "
              "NSDL sector flows and the price-volume footprint.")
    st.info("**What is real and what is estimated.** Named participants, ₹ amounts and prices come from *disclosed* bulk/block deals (large trades only) and the files you upload. "
            "The market does **not** publish how long anyone will hold: the hold windows below combine the *observed minimum* hold (FIFO-matched lots, shareholding streaks, months in an AMC portfolio) "
            "with the *typical length of past accumulation phases of the same stock* (inferred from price × volume). Treat them as research estimates, not disclosures.")
    _uploads()

    c1, c2 = st.columns([1, 2])
    with c1:
        win = st.selectbox("Look-back window", list(WINDOWS), index=1, key="inst_win")
    days = WINDOWS[win]
    with st.spinner("Loading NSE bulk/block deals and the stock-universe scan…"):
        deals, status = ID.deals(days)
        scan = D.scan_master()
    foot_map = scan.get("footprint", {}) or {}
    with c2:
        st.caption(f"Deal source: **{status}** · {len(deals):,} disclosed deals in window · footprint available for {sum(1 for v in foot_map.values() if v.get('ok'))} stocks")

    _cat_metrics(deals)
    tabs = st.tabs(["👥 Who is buying / selling", "🗺️ Sector → stock money map", "⏳ Hold-duration board", "🏦 Fund houses (MF)", "🔎 Stock drill-down"])
    with tabs[0]:
        _tab_who(deals, days)
    with tabs[1]:
        _tab_sector(deals, scan, foot_map)
    with tabs[2]:
        _tab_hold(deals, scan, foot_map)
    with tabs[3]:
        _tab_mf()
    with tabs[4]:
        _tab_stock(deals, scan, foot_map)

    st.caption("Classification of participants into FII/FPI, MF, insurer, DII and others is based on the client name in the exchange disclosure and can be wrong for unusual names. "
               "Bulk/block deals show only trades above exchange thresholds - smaller buying and selling by the same participant is not visible. Not investment advice.")


def _tab_who(deals, days):
    if deals.empty:
        st.warning("No disclosed deals available right now (NSE blocks many cloud servers). Upload NSE's bulk/block deals CSV in the box above, or press 🔄 Refresh Terminal and retry.")
        return
    cat = st.selectbox("Participant type", ["All"] + IE.CATEGORIES, key="inst_cat")
    board = IE.participant_board(deals, cat)
    if board.empty:
        st.info("No deals for this participant type in the window.")
        return
    cfg = {"Net ₹Cr": st.column_config.NumberColumn("Net ₹Cr", format="%.1f"), "Last deal": st.column_config.DateColumn("Last deal", format="DD MMM YYYY")}
    a, b = st.columns(2)
    with a:
        st.markdown("**🟢 Biggest net buyers**")
        st.dataframe(board[board["Net ₹Cr"] > 0].head(25), hide_index=True, column_config=cfg, **STRETCH)
    with b:
        st.markdown("**🔴 Biggest net sellers**")
        st.dataframe(board[board["Net ₹Cr"] < 0].sort_values("Net ₹Cr").head(25), hide_index=True, column_config=cfg, **STRETCH)

    st.markdown("**Sector × participant type - net ₹ Crore (disclosed deals)**")
    sm = IE.sector_matrix(deals)
    nsdl = _nsdl_by_sector()
    if nsdl:
        sm.insert(1, "FPI net ₹Cr (NSDL fortnight)", sm["Sector"].map(nsdl))
    st.dataframe(sm, hide_index=True, **STRETCH)
    st.bar_chart(sm.set_index("Sector")[IE.CATEGORIES], **STRETCH)

    st.markdown("**Stock-wise net ₹ Crore and who is on each side**")
    sa = IE.stock_activity(deals)
    a, b = st.columns(2)
    with a:
        st.caption("Most net bought")
        st.dataframe(sa.head(20).drop(columns=["Deals"]), hide_index=True, **STRETCH)
    with b:
        st.caption("Most net sold")
        st.dataframe(sa.sort_values("Net ₹Cr").head(20).drop(columns=["Deals"]), hide_index=True, **STRETCH)

    with st.expander("All disclosed deals in the window (named)"):
        st.dataframe(deals.drop(columns=["Client"]).head(500), hide_index=True, **STRETCH)
        st.download_button("⬇️ Download deals (CSV)", deals.to_csv(index=False).encode("utf-8"), "institutional_deals.csv", "text/csv", key="inst_deals_dl")


def _nsdl_by_sector():
    try:
        fpi_df, periods, _ = D.fpi_sector()
        if not D.fpi_live(periods):
            return {}
        out = {}
        for _, r in fpi_df.iterrows():
            sec = D.map_nsdl_sector(r["Sector"])
            if sec:
                out[sec] = out.get(sec, 0.0) + float(r["Latest Fortnight"])
        return out
    except Exception:
        return {}


def _tab_sector(deals, scan, foot_map):
    nsdl = _nsdl_by_sector()
    summ = IE.sector_summary(foot_map, deals, SECTOR_STOCKS, nsdl)
    if summ.empty:
        st.info("Sector footprint is not available yet - the stock scan returned no volume data.")
        return
    st.markdown("**Sector money map** - real columns: NSDL FPI flow and disclosed deals; inferred columns: price-volume money-flow proxy and phase length.")
    st.dataframe(summ, hide_index=True, **STRETCH)
    if not nsdl:
        st.caption("NSDL fortnightly sector flow is unreachable right now, so that column is empty (no illustrative numbers are shown here).")
    sec = st.selectbox("Open a sector", list(SECTOR_STOCKS), key="inst_sector")
    sa = IE.stock_activity(deals).set_index("Symbol") if len(deals) else pd.DataFrame()
    rows = []
    for s in SECTOR_STOCKS[sec]:
        fp, r = foot_map.get(s) or {}, scan["results"].get(s)
        if not r:
            continue
        a = sa.loc[s] if len(sa) and s in sa.index else None
        rows.append({"Stock": s, "Name": STOCK_NAMES.get(s, s), "LTP": round(r["price"], 2), "1D %": round(r["chg"], 2),
                     "Deals net ₹Cr": None if a is None else a["Net ₹Cr"], "Top buyer": "-" if a is None else a["Top buyer"], "Top seller": "-" if a is None else a["Top seller"],
                     "Money-flow 5d ₹Cr": fp.get("flow5_cr"), "21d ₹Cr": fp.get("flow21_cr"), "63d ₹Cr": fp.get("flow63_cr"),
                     "Phase": fp.get("phase", "n/a"), "Phase age (sessions)": fp.get("age"), "Estimated hold window": IE.window_text(fp)})
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, **STRETCH)
    st.caption("Money-flow columns are a price × volume proxy for ALL participants (not only institutions). The hold window compares the current phase age with the lengths of earlier phases of the same stock.")


def _tab_hold(deals, scan, foot_map):
    st.markdown("**A. Disclosed positions of named participants - observed minimum holding**")
    fifo = IE.fifo_positions(deals)
    if fifo.empty:
        st.info("No disclosed deals in the window, so no named positions can be traced. Try a longer window or upload NSE's deals CSV.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            cat = st.selectbox("Participant type", ["All"] + IE.CATEGORIES, key="hold_cat")
        with c2:
            state = st.selectbox("Position state", ["Holding", "Exited (matched lots)", "Selling older holding", "All"], key="hold_state")
        f = fifo if cat == "All" else fifo[fifo["Category"] == cat]
        f = f if state == "All" else f[f["State"] == state]
        st.dataframe(f.sort_values("Open ₹Cr", ascending=False).head(200), hide_index=True, **STRETCH)
        st.caption("'Days held (min)' = days since the earliest disclosed buy that has no disclosed sale after it. A smaller, undisclosed sale may exist. "
                   "'Avg closed hold' = real holding period of lots that were bought and later sold inside the window.")
    st.markdown("**B. Typical accumulation / distribution length of each stock (inferred from price × volume)**")
    rows = []
    for s, fp in foot_map.items():
        if not fp.get("ok") or fp["phase"] == "Neutral":
            continue
        w = fp.get("window")
        rows.append({"Stock": s, "Sector": sector_of(s) or "-", "Phase": fp["phase"], "Phase age (sessions)": fp["age"],
                     "Past phases (n)": len(fp["episodes"][fp["phase"]]), "Typical length (25-75%)": f"{w['q25']:.0f}-{w['q75']:.0f}" if w else "n/a",
                     "Median": w["q50"] if w else None, "Est. sessions left": f"{w['rem_low']}-{w['rem_high']}" if w and not w["mature"] else ("fading" if w else "n/a"),
                     "21d money-flow ₹Cr": fp["flow21_cr"], "Reading": IE.window_text(fp)})
    if rows:
        df = pd.DataFrame(rows)
        pick = st.radio("Show", ["Accumulation", "Distribution"], horizontal=True, key="hold_phase")
        st.dataframe(df[df["Phase"] == pick].sort_values("21d money-flow ₹Cr", ascending=(pick == "Distribution")), hide_index=True, **STRETCH)
    else:
        st.info("No stock is in a clear accumulation / distribution phase right now.")


def _tab_mf():
    mf = st.session_state.get("inst_upload_mf")
    if mf is None or len(mf) == 0:
        st.info("Upload one or more AMC monthly portfolio files (box at the top) to see which **fund house** is adding or cutting which stock, how much ₹ and for how many months it has held it. "
                "Two or more consecutive months are needed to show buying / selling.")
        return
    ch = IE.mf_changes(mf)
    months = sorted(mf["Month"].unique())
    st.caption(f"{mf['Fund house'].nunique()} fund house(s) · months loaded: {', '.join(pd.Timestamp(m).strftime('%b %Y') for m in months)}")
    houses = ["All"] + sorted(mf["Fund house"].unique())
    h = st.selectbox("Fund house", houses, key="mf_house")
    v = ch if h == "All" else ch[ch["Fund house"] == h]
    a, b = st.columns(2)
    with a:
        st.markdown("**🟢 Added / new**")
        st.dataframe(v[v["Action"].isin(["ADDED", "NEW"])].sort_values("Net ₹Cr bought / (sold)", ascending=False).head(30), hide_index=True, **STRETCH)
    with b:
        st.markdown("**🔴 Trimmed / exited**")
        st.dataframe(v[v["Action"].isin(["TRIMMED", "EXITED"])].sort_values("Net ₹Cr bought / (sold)").head(30), hide_index=True, **STRETCH)
    st.markdown("**Fund house × sector holding value (₹ Cr, latest month)**")
    last = mf[mf["Month"] == months[-1]]
    st.dataframe(last.pivot_table(index="Fund house", columns="Sector", values="Value_Cr", aggfunc="sum", fill_value=0).round(0).reset_index(), hide_index=True, **STRETCH)
    st.caption("Net ₹ Cr = change in quantity × latest price (price moves are excluded). Months held counts consecutive portfolio months up to the latest one.")


def _tab_stock(deals, scan, foot_map):
    q = st.text_input("Stock symbol", value=st.session_state.get("inst_stock", "RELIANCE"), key="inst_stock_in").strip().upper()
    if not q:
        return
    st.session_state["inst_stock"] = q
    sym = q.replace(".NS", "")
    fp = foot_map.get(sym)
    if not fp:
        _, px = D.hist(sym + ".NS")
        fp = IE.footprint_stats(E.ind(px)) if px is not None and len(px) >= 120 else {"ok": False}
    fifo = IE.fifo_positions(deals[deals["Symbol"] == sym]) if len(deals) else pd.DataFrame()
    shs = st.session_state.get("inst_upload_sh", {}).get(sym)
    sh_ch = IE.shareholding_changes(shs, ID.market_cap_cr(sym)) if shs is not None else None
    mf_all = st.session_state.get("inst_upload_mf")
    mfc = IE.mf_changes(mf_all) if mf_all is not None else None

    st.markdown(f"### {sym} · {STOCK_NAMES.get(sym, '')} · {sector_of(sym) or ''}")
    if fp.get("ok"):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Footprint phase", fp["phase"], f"day {fp['age']}" if fp["age"] else None)
        c2.metric("Money-flow 5d", f"₹{fp['flow5_cr']:+,.0f} Cr")
        c3.metric("Money-flow 21d", f"₹{fp['flow21_cr']:+,.0f} Cr")
        c4.metric("Money-flow 63d", f"₹{fp['flow63_cr']:+,.0f} Cr")
    st.markdown("**Who, how much, for how long**")
    lines = IE.hold_lines(sym, fifo, fp, sh_ch, mfc)
    if lines:
        for ln in lines:
            st.markdown("- " + ln)
    else:
        st.info("No disclosed deals, shareholding or fund-house data for this stock yet. Upload files above, or widen the window.")

    sd = deals[deals["Symbol"] == sym] if len(deals) else pd.DataFrame()
    if len(sd):
        st.markdown("**Disclosed deals (named)**")
        st.dataframe(sd[["Date", "Participant", "Category", "Side", "Qty", "Price", "Value_Cr", "Type"]].rename(columns={"Value_Cr": "₹ Cr"}), hide_index=True, **STRETCH)
    if sh_ch is not None and len(sh_ch):
        st.markdown("**Shareholding pattern changes (uploaded)**")
        st.dataframe(sh_ch, hide_index=True, **STRETCH)
    if mfc is not None and len(mfc[mfc["Stock"] == sym]):
        st.markdown("**Fund houses holding it (uploaded AMC files)**")
        st.dataframe(mfc[mfc["Stock"] == sym], hide_index=True, **STRETCH)
    with st.expander("Yahoo Finance holder tables (best effort - coverage of Indian stocks is patchy)"):
        y = ID.yahoo_holders(sym)
        shown = False
        for key, label in (("institutional", "Institutional holders"), ("mutualfund", "Mutual-fund holders"), ("major", "Major holders")):
            if y.get(key) is not None:
                shown = True
                st.markdown(f"**{label}**")
                st.dataframe(y[key], **STRETCH)
        if not shown:
            st.caption("Yahoo has no holder tables for this symbol.")
