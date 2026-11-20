"""
inst_data.py - network + cache layer for the Institutional Holdings Radar.

Sources, in the order the page trusts them
  1. NSE bulk + block deals  (named client, side, qty, price)  - fetched live, or uploaded as the CSV downloaded from NSE
  2. AMC monthly portfolio files and shareholding-pattern files - uploaded by the user (the only dependable route for
     fund-house level and quarterly category data: no free, stable, official API exists for them)
  3. Yahoo Finance holder tables (best effort; coverage of Indian stocks is patchy)
Nothing here ever invents numbers: when a source is unreachable the function returns an empty frame and a status string.
"""
from __future__ import annotations

import io
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import pandas as pd
import streamlit as st
import yfinance as yf

import data as D
import inst_engine as IE
from knowledge import SECTOR_STOCKS


def _fmt(d: date) -> str:
    return d.strftime("%d-%m-%Y")


def _rows(payload):
    return (payload.get("data") if isinstance(payload, dict) else payload) or []


@st.cache_data(ttl=1800, show_spinner=False)
def _deals_window_cached(days: int):
    """All-symbol bulk + block deals for the last `days` days, fetched in <=30-day windows."""
    to_d = date.today()
    cur, rows = to_d - timedelta(days=days), []
    while cur <= to_d:
        end = min(cur + timedelta(days=29), to_d)
        for kind in ("bulk-deals", "block-deals"):
            try:
                for r in _rows(D.nse_api(f"historical/{kind}?from={_fmt(cur)}&to={_fmt(end)}")):
                    rows.append({**r, "_kind": "Bulk" if kind == "bulk-deals" else "Block"})
            except Exception:
                continue
        cur = end + timedelta(days=1)
    df = D._norm_deals(rows)
    if len(df):
        df["Type"] = [r["_kind"] for r in rows]
    return df


@st.cache_data(ttl=1800, show_spinner=False)
def _deals_per_symbol_cached(days: int, budget_s: int = 75):
    """Fallback when the all-symbol window is empty: per-stock history for the sector universe (time boxed)."""
    syms = sorted({s for v in SECTOR_STOCKS.values() for s in v})
    t0, out = time.time(), []

    def one(s):
        if time.time() - t0 > budget_s:
            return None
        try:
            return D.deals_for_symbol(s, days)
        except Exception:
            return None
    with ThreadPoolExecutor(max_workers=4) as ex:
        for res in ex.map(one, syms):
            if res is not None and len(res):
                out.append(res)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def live_deals(days: int):
    """-> (normalised deals frame, status text).  Failed / empty fetches are not left in the cache."""
    status = "NSE all-symbol deal history"
    raw = _deals_window_cached(days)
    if raw is None or len(raw) == 0:
        _deals_window_cached.clear()
        raw, status = _deals_per_symbol_cached(days), "NSE per-stock deal history (sector universe)"
        if raw is None or len(raw) == 0:
            _deals_per_symbol_cached.clear()
            return IE.normalise_deals(None), "unavailable - NSE did not answer from this server (upload NSE's deals CSV instead)"
    return IE.normalise_deals(raw), status


def deals(days: int):
    """Live deals merged with anything the user uploaded; filtered to the window."""
    live, status = live_deals(days)
    up = st.session_state.get("inst_upload_deals")
    frames = [x for x in (live, up) if x is not None and len(x)]
    if not frames:
        return IE.normalise_deals(None), status
    allv = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["Date", "Symbol", "Client", "Side", "Qty", "Price"])
    if up is not None and len(up):
        status += " + uploaded file"
    return IE.filter_days(allv, days), status


# ------------------------------------------------------------------------------
# Uploads
# ------------------------------------------------------------------------------
def read_any(file) -> dict[str, pd.DataFrame]:
    """CSV or Excel upload -> {sheet name: raw frame with header=None}."""
    name = getattr(file, "name", "file").lower()
    data = file.getvalue() if hasattr(file, "getvalue") else file.read()
    if name.endswith(".csv"):
        return {"csv": pd.read_csv(io.BytesIO(data), header=None, dtype=str, keep_default_na=False)}
    return pd.read_excel(io.BytesIO(data), sheet_name=None, header=None, dtype=str)


def csv_with_header(file) -> pd.DataFrame:
    data = file.getvalue() if hasattr(file, "getvalue") else file.read()
    name = getattr(file, "name", "f").lower()
    return pd.read_csv(io.BytesIO(data)) if name.endswith(".csv") else pd.read_excel(io.BytesIO(data))


def guess_month(filename: str):
    """'SBI_Portfolio_Oct2026.xlsx' / 'portfolio-2026-10.csv' -> Timestamp, else today's month."""
    f = filename.lower()
    m = re.search(r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s_\-\.]*'?(\d{2,4})", f)
    if m:
        y = int(m.group(2))
        return pd.Timestamp(year=2000 + y if y < 100 else y, month=["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"].index(m.group(1)) + 1, day=1)
    m = re.search(r"(20\d{2})[\-_\.](\d{1,2})", f)
    if m and 1 <= int(m.group(2)) <= 12:
        return pd.Timestamp(year=int(m.group(1)), month=int(m.group(2)), day=1)
    return pd.Timestamp.today().to_period("M").to_timestamp()


def guess_fund_house(filename: str) -> str:
    base = re.sub(r"\.[a-z]+$", "", filename, flags=re.I)
    base = re.sub(r"(portfolio|holdings|monthly|disclosure|statement|20\d{2}|\b\d{1,2}\b|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*", " ", base, flags=re.I)
    return IE._title(re.sub(r"[_\-\.]+", " ", base)).strip() or "Unnamed fund house"


# ------------------------------------------------------------------------------
# Yahoo holder tables and market cap
# ------------------------------------------------------------------------------
@st.cache_data(ttl=21600, show_spinner=False)
def yahoo_holders(symbol: str) -> dict:
    out = {"major": None, "institutional": None, "mutualfund": None}
    try:
        t = yf.Ticker(symbol.upper() + ".NS")
    except Exception:
        return out
    for key, attr in (("major", "major_holders"), ("institutional", "institutional_holders"), ("mutualfund", "mutualfund_holders")):
        try:
            v = getattr(t, attr)
            if v is not None and len(v):
                out[key] = v
        except Exception:
            continue
    return out


@st.cache_data(ttl=21600, show_spinner=False)
def market_cap_cr(symbol: str):
    try:
        mc = (D.info(symbol.upper() + ".NS") or {}).get("marketCap")
        return float(mc) / 1e7 if mc else None
    except Exception:
        return None
