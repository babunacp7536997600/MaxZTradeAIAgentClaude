"""
inst_engine.py - pure analytics for big-player (FII/FPI, MF, insurance, DII, prop/HNI) activity.

No Streamlit and no network here, so everything is unit-testable offline.

What is REAL (disclosed) and what is INFERRED - the module keeps the two apart:

REAL / disclosed
  * NSE bulk + block deals: named client, side, quantity, price  (large trades only, same-day disclosure)
  * Quarterly shareholding pattern (uploaded or fetched): FII / DII / MF / Insurance / Promoter %
  * AMC monthly portfolio disclosures (uploaded): which fund house holds how much of which stock
INFERRED (price x volume footprint, all participants combined - NOT an institutional disclosure)
  * money-flow proxy in Rs Cr, accumulation / distribution phase, typical phase length

"How many days will they hold" is never published.  We therefore report
  (a) the OBSERVED minimum hold (FIFO-matched bulk/block lots, shareholding streaks, MF months held) and
  (b) the typical length of past accumulation / distribution phases of the same stock (inferred).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from knowledge import STOCK_NAMES, sector_of

EPS = 1e-9
CATEGORIES = ["FII/FPI", "Mutual Fund", "Insurance", "Domestic Institution", "Prop / HNI / Other"]

# ------------------------------------------------------------------------------
# Participant classification (name based - a heuristic, shown as such in the UI)
# ------------------------------------------------------------------------------
_FOREIGN = (
    "morgan stanley", "goldman sachs", "societe generale", "citigroup", "jpmorgan", "j p morgan", "bnp paribas", "nomura",
    "vanguard", "blackrock", "ishares", "abu dhabi", "government of singapore", "norges", "fidelity", "merrill lynch",
    "barclays", "deutsche", "credit suisse", "hsbc", "jefferies", "macquarie", "copthall", "mauritius", "singapore",
    "luxembourg", "cayman", "ireland", "emerging market", "capital group", "t. rowe", "t rowe", "wellington", "schroder",
    "amundi", "bofa", "bank of america", "state street", "nordea", "abrdn", "aberdeen", "pictet", "lazard", "oppenheimer",
    "jane street", "optiver", "citadel", "foreign", "kuwait investment", "saudi", "qatar", "temasek",
    "ontario teachers", "canada pension", "calpers", "california public", "stichting", "limited partnership",
    "norwest", "gaoling", "mirae asset global", "nikko",
)
_FOREIGN_RX = re.compile(r"\b(?:odi|fpi|pte|plc|llc|gmbh|gic|ubs|l\.p)\b", re.I)
_MF_RX = re.compile(r"mutual fund|asset management|\betf\b|\bamc\b|index fund|(?:trustee|trusteeship).*(?:a/c|fund)|(?:a/c|fund).*(?:trustee|trusteeship)", re.I)
_INS_RX = re.compile(r"life insurance|general insurance|\blic\b|insurance co|assurance|sbi life|hdfc life|icici prudential life|icici lombard|max life|"
                     r"bajaj allianz|new india assur|national insurance|oriental insurance|united india insurance|star health|tata aia|kotak life", re.I)
_DOM_RX = re.compile(r"provident fund|pension|epfo|nps trust|national pension|\bbank\b|alternative investment|\baif\b|venture|"
                     r"investment fund|family office|sovereign|government of india|\bcoal india\b.*trust", re.I)


_ACR = {"HDFC", "ICICI", "IDFC", "SBI", "LIC", "UTI", "DSP", "BNP", "HSBC", "GIC", "PTE", "LLC", "PLC", "LLP", "ETF", "FPI", "ODI",
        "NPS", "EPFO", "AIF", "ABN", "AMRO", "ASIA", "UBS", "JP", "IIFL", "HSBC", "BOFA", "AXA", "AIA", "PNB", "SG"}


def _title(s: str) -> str:
    def fix(m):
        w = m.group(0)
        if w.upper() in _ACR and w.upper() not in ("ASIA",) or (w.isupper() and len(w) <= 3):
            return w.upper()
        return w.capitalize()
    return re.sub(r"[A-Za-z&']+", fix, str(s).strip())


def classify_client(name: str) -> str:
    """Name-based category.  Order matters: MF -> insurance -> foreign -> domestic institution -> other."""
    n = re.sub(r"\s+", " ", str(name or "")).strip()
    low = n.lower() + " "
    if not n:
        return CATEGORIES[-1]
    if _MF_RX.search(n):
        return "Mutual Fund"
    if _INS_RX.search(n):
        return "Insurance"
    if any(k in low for k in _FOREIGN) or _FOREIGN_RX.search(n):
        return "FII/FPI"
    if _DOM_RX.search(n):
        return "Domestic Institution"
    return "Prop / HNI / Other"


def amc_of(name: str) -> str:
    """'SBI MUTUAL FUND - SBI CONTRA FUND' -> 'SBI Mutual Fund'; '...TRUSTEE LTD- A/C X FUND' -> trustee's AMC."""
    n = re.sub(r"\s+", " ", str(name or "")).strip()
    m = re.search(r"^(.*?mutual fund)", n, re.I)
    if m:
        return _title(m.group(1))
    m = re.search(r"^(.*?)(?:\s*-\s+|\s*-a/c|\s+a/c\b)", n, re.I)
    base = m.group(1) if m else n
    base = re.sub(r"\b(trustee|trusteeship|ltd|limited|pvt|private|company|co)\b\.?", "", base, flags=re.I)
    base = re.sub(r"\s+", " ", base).strip(" -.")
    return _title(base) or _title(n)


def participant_name(name: str, category: str) -> str:
    n = re.sub(r"\s+", " ", str(name or "")).strip()
    return amc_of(n) if category == "Mutual Fund" else _title(n)


# ------------------------------------------------------------------------------
# Bulk / block deals
# ------------------------------------------------------------------------------
DEAL_COLS = ["Date", "Symbol", "Sector", "Client", "Participant", "Category", "Side", "Qty", "Price", "Value_Cr", "Signed_Cr", "Type"]


def _find(df, *keys):
    for c in df.columns:
        lc = str(c).lower()
        if any(k in lc for k in keys):
            return c
    return None


def _num(s):
    return pd.to_numeric(s.astype(str).str.replace(r"[,₹\s]", "", regex=True), errors="coerce")


def normalise_deals(df: pd.DataFrame | None) -> pd.DataFrame:
    """Any NSE-style deals table (API rows or the CSV downloaded from NSE) -> clean frame with category + Rs Cr."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=DEAL_COLS)
    c_date, c_sym = _find(df, "date"), _find(df, "symbol")
    c_cli, c_side = _find(df, "client"), _find(df, "buy")
    c_qty, c_px = _find(df, "qty", "quantity"), _find(df, "watp", "price")
    if None in (c_date, c_sym, c_cli, c_side, c_qty, c_px):
        return pd.DataFrame(columns=DEAL_COLS)
    c_typ = _find(df, "type")
    out = pd.DataFrame({
        "Date": pd.to_datetime(df[c_date], errors="coerce", dayfirst=True, format="mixed"),
        "Symbol": df[c_sym].astype(str).str.upper().str.strip(),
        "Client": df[c_cli].astype(str).str.strip(),
        "Side": df[c_side].astype(str).str.upper().str.strip().str[:1].map({"B": "BUY", "S": "SELL"}),
        "Qty": _num(df[c_qty]), "Price": _num(df[c_px]),
        "Type": df[c_typ].astype(str) if c_typ is not None else "Deal",
    }).dropna(subset=["Date", "Side", "Qty", "Price"])
    out = out[(out["Qty"] > 0) & (out["Price"] > 0) & (out["Symbol"] != "")]
    if out.empty:
        return pd.DataFrame(columns=DEAL_COLS)
    out["Category"] = out["Client"].map(classify_client)
    out["Participant"] = [participant_name(c, k) for c, k in zip(out["Client"], out["Category"])]
    out["Sector"] = out["Symbol"].map(lambda s: sector_of(s) or "Other / unmapped")
    out["Value_Cr"] = out["Qty"] * out["Price"] / 1e7
    out["Signed_Cr"] = np.where(out["Side"] == "BUY", out["Value_Cr"], -out["Value_Cr"])
    out = out.drop_duplicates(subset=["Date", "Symbol", "Client", "Side", "Qty", "Price"])
    return out[DEAL_COLS].sort_values("Date", ascending=False).reset_index(drop=True)


def filter_days(deals: pd.DataFrame, days: int, asof=None) -> pd.DataFrame:
    if deals.empty:
        return deals
    asof = pd.Timestamp(asof) if asof is not None else pd.Timestamp.today().normalize()
    return deals[deals["Date"] >= asof - pd.Timedelta(days=int(days))]


def _fmt_list(series: pd.Series, n: int = 3) -> str:
    return ", ".join(f"{k} ({v:,.0f})" for k, v in series.head(n).items()) or "-"


def participant_board(deals: pd.DataFrame, category: str | None = None) -> pd.DataFrame:
    """One row per named participant: amount bought / sold (Rs Cr), and where the money is going."""
    cols = ["Participant", "Category", "Bought ₹Cr", "Sold ₹Cr", "Net ₹Cr", "Deals", "Stocks being bought (₹Cr)",
            "Stocks being sold (₹Cr)", "Sectors bought (₹Cr)", "Sectors sold (₹Cr)", "Last deal"]
    if deals.empty:
        return pd.DataFrame(columns=cols)
    d = deals if category in (None, "All") else deals[deals["Category"] == category]
    rows = []
    for (p, c), g in d.groupby(["Participant", "Category"]):
        by_stk = g.groupby("Symbol")["Signed_Cr"].sum()
        by_sec = g.groupby("Sector")["Signed_Cr"].sum()
        buy, sell = g.loc[g["Side"] == "BUY", "Value_Cr"].sum(), g.loc[g["Side"] == "SELL", "Value_Cr"].sum()
        rows.append({
            "Participant": p, "Category": c, "Bought ₹Cr": round(buy, 1), "Sold ₹Cr": round(sell, 1), "Net ₹Cr": round(buy - sell, 1),
            "Deals": int(len(g)),
            "Stocks being bought (₹Cr)": _fmt_list(by_stk[by_stk > 0].sort_values(ascending=False)),
            "Stocks being sold (₹Cr)": _fmt_list((-by_stk[by_stk < 0]).sort_values(ascending=False)),
            "Sectors bought (₹Cr)": _fmt_list(by_sec[by_sec > 0].sort_values(ascending=False), 2),
            "Sectors sold (₹Cr)": _fmt_list((-by_sec[by_sec < 0]).sort_values(ascending=False), 2),
            "Last deal": g["Date"].max(),
        })
    return pd.DataFrame(rows, columns=cols).sort_values("Net ₹Cr", ascending=False).reset_index(drop=True)


def sector_matrix(deals: pd.DataFrame) -> pd.DataFrame:
    """Sector x participant-category net Rs Cr from disclosed deals."""
    if deals.empty:
        return pd.DataFrame(columns=["Sector"] + CATEGORIES + ["Total ₹Cr"])
    p = deals.pivot_table(index="Sector", columns="Category", values="Signed_Cr", aggfunc="sum", fill_value=0.0)
    for c in CATEGORIES:
        if c not in p.columns:
            p[c] = 0.0
    p = p[CATEGORIES].round(1)
    p["Total ₹Cr"] = p.sum(axis=1).round(1)
    return p.sort_values("Total ₹Cr", ascending=False).reset_index()


def stock_activity(deals: pd.DataFrame) -> pd.DataFrame:
    """One row per stock: net Rs Cr by category and the biggest named buyer / seller."""
    cols = ["Symbol", "Sector", "Net ₹Cr"] + CATEGORIES + ["Top buyer", "Top seller", "Deals", "Last deal"]
    if deals.empty:
        return pd.DataFrame(columns=cols)
    rows = []
    for s, g in deals.groupby("Symbol"):
        by_p = g.groupby("Participant")["Signed_Cr"].sum()
        by_c = g.groupby("Category")["Signed_Cr"].sum()
        row = {"Symbol": s, "Sector": g["Sector"].iloc[0], "Net ₹Cr": round(float(g["Signed_Cr"].sum()), 1),
               "Top buyer": by_p.idxmax() if by_p.max() > 0 else "-", "Top seller": by_p.idxmin() if by_p.min() < 0 else "-",
               "Deals": int(len(g)), "Last deal": g["Date"].max()}
        for c in CATEGORIES:
            row[c] = round(float(by_c.get(c, 0.0)), 1)
        rows.append(row)
    return pd.DataFrame(rows, columns=cols).sort_values("Net ₹Cr", ascending=False).reset_index(drop=True)


def fifo_positions(deals: pd.DataFrame, asof=None) -> pd.DataFrame:
    """
    FIFO-match each participant's disclosed buys against its later disclosed sells in the same stock.

    Observed facts only: lots still open are 'held at least N days' (a smaller undisclosed sale may exist),
    closed lots give a real holding period, sells with no earlier buy in the window are 'older holding being sold'.
    """
    cols = ["Participant", "Category", "Symbol", "Sector", "State", "Open qty", "Open ₹Cr", "Open since", "Days held (min)",
            "Closed lots", "Avg closed hold (days)", "Older stock sold (qty)", "Net ₹Cr"]
    if deals.empty:
        return pd.DataFrame(columns=cols)
    asof = pd.Timestamp(asof) if asof is not None else pd.Timestamp.today().normalize()
    rows = []
    for (p, s), g in deals.sort_values("Date", kind="stable").groupby(["Participant", "Symbol"]):
        lots, closed_q, closed_w, closed_n, older = [], 0.0, 0.0, 0, 0.0
        for _, r in g.iterrows():
            if r["Side"] == "BUY":
                lots.append([r["Date"], float(r["Qty"]), float(r["Price"])])
                continue
            need = float(r["Qty"])
            while need > 1e-9 and lots:
                take = min(need, lots[0][1])
                days = max((r["Date"] - lots[0][0]).days, 0)
                closed_q += take
                closed_w += take * days
                closed_n += 1
                lots[0][1] -= take
                need -= take
                if lots[0][1] <= 1e-9:
                    lots.pop(0)
            older += max(need, 0.0)
        open_q = sum(l[1] for l in lots)
        last_px = float(g.iloc[-1]["Price"])
        since = min((l[0] for l in lots), default=pd.NaT)
        state = ("Holding" if open_q > 0 else "Exited (matched lots)" if closed_q > 0 else "Selling older holding" if older > 0 else "-")
        rows.append({
            "Participant": p, "Category": g["Category"].iloc[0], "Symbol": s, "Sector": g["Sector"].iloc[0], "State": state,
            "Open qty": round(open_q), "Open ₹Cr": round(open_q * last_px / 1e7, 2), "Open since": since,
            "Days held (min)": (int((asof - since).days) if pd.notna(since) else None), "Closed lots": closed_n,
            "Avg closed hold (days)": (round(closed_w / closed_q, 1) if closed_q > 0 else None),
            "Older stock sold (qty)": round(older), "Net ₹Cr": round(float(g["Signed_Cr"].sum()), 2),
        })
    return pd.DataFrame(rows, columns=cols).sort_values("Net ₹Cr", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------------------
# Price x volume footprint (INFERRED) and typical phase length
# ------------------------------------------------------------------------------
def _runs(mask, max_gap: int = 2, min_len: int = 5):
    """True-runs of `mask`; gaps up to `max_gap` bars are bridged, runs shorter than `min_len` are dropped."""
    m = np.asarray(mask, bool)
    n, i, runs = len(m), 0, []
    while i < n:
        if not m[i]:
            i += 1
            continue
        start = last = i
        gap, j = 0, i + 1
        while j < n:
            if m[j]:
                last, gap = j, 0
            else:
                gap += 1
                if gap > max_gap:
                    break
            j += 1
        if last - start + 1 >= min_len:
            runs.append((start, last))
        i = last + 1
    return runs


def _window(lengths, age: int):
    if len(lengths) < 4:
        return None
    q25, q50, q75 = (float(x) for x in np.percentile(lengths, [25, 50, 75]))
    return {"n": len(lengths), "q25": q25, "q50": q50, "q75": q75, "age": int(age),
            "rem_low": max(int(round(q25 - age)), 0), "rem_high": max(int(round(q75 - age)), 1), "mature": bool(age > q75)}


def footprint_stats(d: pd.DataFrame | None, min_len: int = 5) -> dict:
    """
    From an `engine.ind()` frame: money-flow proxy (Rs Cr), current accumulation / distribution phase, its age, and how long
    previous phases of the SAME stock lasted.  Inferred from price x volume; it is not who-bought-what.
    """
    if d is None or len(d) < 120 or "HAS_VOL" not in d or not bool(d["HAS_VOL"].iloc[-1]):
        return {"ok": False}
    n = len(d)
    cmf, obv = d["CMF20"].fillna(0).values, d["OBV_SLOPE"].fillna(0).values
    acc, dist = (cmf > 0.05) & (obv > 0), (cmf < -0.05) & (obv < 0)
    tp = (d["High"] + d["Low"] + d["Close"]) / 3
    mfm = ((d["Close"] - d["Low"]) - (d["High"] - d["Close"])) / (d["High"] - d["Low"] + EPS)
    flow = (mfm * d["Volume"].astype(float) * tp / 1e7)
    out = {"ok": True, "price": float(d["Close"].iloc[-1]), "asof": d.index[-1]}
    for k in (5, 21, 63):
        out[f"flow{k}_cr"] = round(float(flow.tail(k).sum()), 1)
    runs = {"Accumulation": _runs(acc, 2, min_len), "Distribution": _runs(dist, 2, min_len)}

    def ongoing(rs):
        return rs[-1] if rs and rs[-1][1] >= n - 3 else None
    cur = {k: ongoing(v) for k, v in runs.items()}
    phase, run = "Neutral", None
    if cur["Accumulation"] and (not cur["Distribution"] or cur["Accumulation"][1] >= cur["Distribution"][1]):
        phase, run = "Accumulation", cur["Accumulation"]
    elif cur["Distribution"]:
        phase, run = "Distribution", cur["Distribution"]
    done = {k: [e - s + 1 for s, e in v if (s, e) != cur[k]] for k, v in runs.items()}
    out.update(phase=phase, age=(run[1] - run[0] + 1) if run else 0, episodes=done,
               avg_cmf=round(float(cmf[run[0]:run[1] + 1].mean()), 3) if run else 0.0)
    out["window"] = _window(done[phase], out["age"]) if run else None
    return out


def window_text(fp: dict) -> str:
    if not fp or not fp.get("ok"):
        return "No volume data - cannot estimate"
    w = fp.get("window")
    if fp["phase"] == "Neutral":
        return "No active accumulation / distribution phase"
    if not w:
        return f"{fp['phase']} for {fp['age']} sessions; too few past phases (<4) for a window"
    tail = "phase already older than 75% of past ones - fading risk" if w["mature"] else f"about {w['rem_low']}-{w['rem_high']} more sessions"
    return f"Day {w['age']} of a phase that usually lasts {w['q25']:.0f}-{w['q75']:.0f} sessions (median {w['q50']:.0f}, n={w['n']}) -> {tail}"


# ------------------------------------------------------------------------------
# Shareholding pattern (uploaded / fetched)
# ------------------------------------------------------------------------------
_CAT_MAP = [("promoter", "Promoter"), ("fii", "FII/FPI"), ("fpi", "FII/FPI"), ("foreign", "FII/FPI"), ("mutual", "Mutual Fund"),
            ("insurance", "Insurance"), ("dii", "DII (total)"), ("domestic", "DII (total)"), ("government", "Government"),
            ("public", "Public / retail"), ("retail", "Public / retail"), ("non-inst", "Public / retail"), ("non inst", "Public / retail")]


def _norm_cat(x: str) -> str:
    low = str(x).lower()
    for k, v in _CAT_MAP:
        if k in low:
            return v
    return str(x).strip()


def parse_shareholding(df: pd.DataFrame | None) -> pd.DataFrame:
    """Long (Quarter, Category, Pct) from long / quarters-as-columns / quarters-as-rows layouts."""
    empty = pd.DataFrame(columns=["Quarter", "Category", "Pct"])
    if df is None or df.shape[0] < 2 or df.shape[1] < 2:
        return empty
    df = df.copy()
    cols = [str(c).lower() for c in df.columns]
    if any("categ" in c for c in cols) and any(("pct" in c or "%" in c or "holding" in c) for c in cols):
        cq = next(c for c in df.columns if any(k in str(c).lower() for k in ("quarter", "date", "period")))
        cc = next(c for c in df.columns if "categ" in str(c).lower())
        cp = next(c for c in df.columns if any(k in str(c).lower() for k in ("pct", "%", "holding")))
        long = pd.DataFrame({"Quarter": df[cq], "Category": df[cc], "Pct": df[cp]})
    else:
        first = pd.to_datetime(df.iloc[:, 0], errors="coerce", format="mixed")
        if first.notna().mean() > 0.6:
            long = df.melt(id_vars=df.columns[0], var_name="Category", value_name="Pct").rename(columns={df.columns[0]: "Quarter"})
        else:
            long = df.melt(id_vars=df.columns[0], var_name="Quarter", value_name="Pct").rename(columns={df.columns[0]: "Category"})
    long["Quarter"] = pd.to_datetime(long["Quarter"], errors="coerce", format="mixed")
    long["Pct"] = pd.to_numeric(long["Pct"].astype(str).str.replace(r"[%,\s]", "", regex=True), errors="coerce")
    long["Category"] = long["Category"].map(_norm_cat)
    long = long.dropna(subset=["Quarter", "Pct"])
    return long.groupby(["Quarter", "Category"], as_index=False)["Pct"].sum().sort_values("Quarter") if len(long) else empty


def shareholding_changes(long: pd.DataFrame, market_cap_cr: float | None = None) -> pd.DataFrame:
    """Latest %, quarter-on-quarter change, streak of consecutive rises / falls, and the Rs Cr that change represents."""
    cols = ["Category", "Latest %", "Prev quarter %", "QoQ change (pp)", "4Q change (pp)", "Streak (quarters)", "Direction",
            "Approx ₹Cr moved (QoQ)", "Quarters of data"]
    if long is None or long.empty:
        return pd.DataFrame(columns=cols)
    p = long.pivot_table(index="Quarter", columns="Category", values="Pct", aggfunc="sum").sort_index()
    rows = []
    for c in p.columns:
        s = p[c].dropna()
        if len(s) < 2:
            continue
        diff = s.diff().dropna()
        sign = np.sign(diff.where(diff.abs() >= 0.005, 0.0)).values
        streak, last = 0, sign[-1]
        for v in sign[::-1]:
            if v == last and v != 0:
                streak += 1
            else:
                break
        rows.append({
            "Category": c, "Latest %": round(float(s.iloc[-1]), 2), "Prev quarter %": round(float(s.iloc[-2]), 2),
            "QoQ change (pp)": round(float(s.iloc[-1] - s.iloc[-2]), 2),
            "4Q change (pp)": round(float(s.iloc[-1] - s.iloc[-5]), 2) if len(s) >= 5 else None,
            "Streak (quarters)": int(streak) * (1 if last > 0 else -1 if last < 0 else 0),
            "Direction": "Adding" if last > 0 else "Reducing" if last < 0 else "Unchanged",
            "Approx ₹Cr moved (QoQ)": round(float(s.iloc[-1] - s.iloc[-2]) / 100 * market_cap_cr, 0) if market_cap_cr else None,
            "Quarters of data": int(len(s)),
        })
    return pd.DataFrame(rows, columns=cols)


# ------------------------------------------------------------------------------
# AMC (fund house) monthly portfolio disclosures
# ------------------------------------------------------------------------------
_NORM_RX = re.compile(r"\b(limited|ltd|corporation|corp|company|co|industries|india|the|of|and|&)\b|[^a-z0-9 ]")


def _norm_name(x: str) -> str:
    return re.sub(r"\s+", " ", _NORM_RX.sub(" ", str(x).lower())).strip()


_NAME_TO_SYM = {_norm_name(v): k for k, v in STOCK_NAMES.items()}


def symbol_from_name(name: str) -> str | None:
    n = _norm_name(name)
    if not n:
        return None
    if n in _NAME_TO_SYM:
        return _NAME_TO_SYM[n]
    hits = [s for k, s in _NAME_TO_SYM.items() if len(k) >= 4 and (k in n or n in k)]
    return hits[0] if len(hits) == 1 else None


def parse_amc_portfolio(raw: pd.DataFrame | None, fund_house: str, month, scheme: str = "") -> pd.DataFrame:
    """
    Equity lines of one AMC portfolio sheet (read with header=None).  Handles the title rows above the table and values
    in Rs Lakhs (SEBI format) or Rs Crore.
    """
    cols = ["Fund house", "Scheme", "Month", "Instrument", "Symbol", "Sector", "Qty", "Value_Cr", "Pct_NAV"]
    if raw is None or raw.empty:
        return pd.DataFrame(columns=cols)
    raw = raw.reset_index(drop=True)
    hdr = None
    for i in range(min(len(raw), 60)):
        row = " | ".join(str(x).lower() for x in raw.iloc[i].tolist())
        if re.search(r"name of (the )?instrument|instrument name|security name|name of security", row):
            hdr = i
            break
    if hdr is None:
        return pd.DataFrame(columns=cols)
    head = [str(x).strip() for x in raw.iloc[hdr].tolist()]
    body = raw.iloc[hdr + 1:].copy()
    body.columns = head
    title_blob = " ".join(str(x).lower() for x in raw.iloc[:hdr + 1].values.ravel())

    def col(*keys):
        for c in head:
            if any(k in c.lower() for k in keys):
                return c
        return None
    c_name = col("name of", "instrument", "security")
    c_isin, c_ind = col("isin"), col("industry", "sector", "rating")
    c_qty, c_val = col("quantity", "qty", "no. of"), col("market", "fair value", "value")
    c_pct = col("% to nav", "% of nav", "to nav", "percent")
    if c_name is None or c_val is None:
        return pd.DataFrame(columns=cols)
    unit = 1 / 100.0  # Rs Lakhs -> Rs Cr (SEBI default)
    vlow = (c_val or "").lower() + " " + title_blob
    if re.search(r"crore|\bcr\b", vlow) and not re.search(r"lakh", vlow):
        unit = 1.0
    body = body.dropna(subset=[c_name])
    if c_isin is not None:
        body = body[body[c_isin].astype(str).str.match(r"^IN[EF]\w{4}01\w{3}$", na=False)]
    if body.empty:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame({
        "Fund house": fund_house, "Scheme": scheme or "All schemes", "Month": pd.Timestamp(month).to_period("M").to_timestamp(),
        "Instrument": body[c_name].astype(str).str.strip(),
        "Qty": _num(body[c_qty]) if c_qty else np.nan, "Value_Cr": _num(body[c_val]) * unit,
        "Pct_NAV": _num(body[c_pct]) if c_pct else np.nan,
    })
    out["Symbol"] = out["Instrument"].map(symbol_from_name)
    ind = body[c_ind].astype(str).str.strip() if c_ind else pd.Series("", index=body.index)
    out["Sector"] = [sector_of(s) if s else (i or "Other / unmapped") for s, i in zip(out["Symbol"], ind)]
    return out.dropna(subset=["Value_Cr"]).reset_index(drop=True)[cols]


def mf_changes(holdings: pd.DataFrame) -> pd.DataFrame:
    """Month-on-month action of each fund house in each stock, with months held so far and the Rs Cr added / cut."""
    cols = ["Fund house", "Stock", "Sector", "Latest month", "Qty (latest)", "Value ₹Cr (latest)", "Action", "Net ₹Cr bought / (sold)", "Months held (streak)"]
    if holdings is None or holdings.empty or holdings["Month"].nunique() < 1:
        return pd.DataFrame(columns=cols)
    h = holdings.copy()
    h["Stock"] = h["Symbol"].fillna(h["Instrument"])
    g = h.groupby(["Fund house", "Stock", "Month"], as_index=False).agg(Qty=("Qty", "sum"), Value_Cr=("Value_Cr", "sum"), Sector=("Sector", "first"))
    months = sorted(g["Month"].unique())
    latest, prev = months[-1], (months[-2] if len(months) > 1 else None)
    rows = []
    for (fh, stk), s in g.groupby(["Fund house", "Stock"]):
        s = s.set_index("Month").sort_index()
        have = [m in s.index and s.loc[m, "Value_Cr"] > 0 for m in months]
        streak = 0
        for v in have[::-1]:
            if v:
                streak += 1
            else:
                break
        cur = s.loc[latest] if latest in s.index else None
        pre = s.loc[prev] if prev is not None and prev in s.index else None
        if cur is None and pre is None:
            continue
        cq = float(cur["Qty"]) if cur is not None and pd.notna(cur["Qty"]) else 0.0
        pq = float(pre["Qty"]) if pre is not None and pd.notna(pre["Qty"]) else 0.0
        px = (float(cur["Value_Cr"]) / cq) if cur is not None and cq > 0 else (float(pre["Value_Cr"]) / pq if pre is not None and pq > 0 else 0.0)
        if prev is None:
            action, net = "Holding (1 month only)", np.nan
        elif cur is None:
            action, net = "EXITED", -pq * px
        elif pre is None:
            action, net = "NEW", cq * px
        else:
            dq = cq - pq
            action = "ADDED" if dq > 0.01 * max(pq, 1) else "TRIMMED" if dq < -0.01 * max(pq, 1) else "UNCHANGED"
            net = dq * px
        ref = cur if cur is not None else pre
        rows.append({"Fund house": fh, "Stock": stk, "Sector": ref["Sector"], "Latest month": latest if cur is not None else prev,
                     "Qty (latest)": round(cq), "Value ₹Cr (latest)": round(float(cur["Value_Cr"]), 1) if cur is not None else 0.0,
                     "Action": action, "Net ₹Cr bought / (sold)": None if pd.isna(net) else round(float(net), 1), "Months held (streak)": streak})
    return pd.DataFrame(rows, columns=cols).sort_values("Net ₹Cr bought / (sold)", ascending=False, na_position="last").reset_index(drop=True)


# ------------------------------------------------------------------------------
# Sector roll-up: how much money, in which sector, for roughly how long
# ------------------------------------------------------------------------------
def sector_summary(foot_map: dict, deals: pd.DataFrame, sector_stocks: dict, nsdl: dict | None = None) -> pd.DataFrame:
    """
    One row per sector.  Columns mix REAL data (NSDL FPI sector flow, disclosed deal net by category) with the INFERRED
    price-volume money-flow proxy and the typical length of accumulation phases of the stocks in the sector.
    """
    dm = sector_matrix(deals).set_index("Sector") if deals is not None and len(deals) else None
    rows = []
    for sec, syms in sector_stocks.items():
        fps = [foot_map[s] for s in syms if foot_map.get(s, {}).get("ok")]
        acc = [f for f in fps if f["phase"] == "Accumulation"]
        dis = [f for f in fps if f["phase"] == "Distribution"]
        flow21 = sum(f["flow21_cr"] for f in fps)
        med = [f["window"]["q50"] for f in acc if f.get("window")]
        row = {"Sector": sec, "FPI net ₹Cr (NSDL, fortnight)": (nsdl or {}).get(sec),
               "Disclosed deals net ₹Cr": None, **{f"Deals: {c}": None for c in CATEGORIES[:4]},
               "Inferred flow 21d ₹Cr": round(flow21, 0) if fps else None, "Accumulating": len(acc), "Distributing": len(dis),
               "Stocks tracked": len(fps), "Typical accumulation (sessions)": round(float(np.median(med)), 0) if med else None}
        if dm is not None and sec in dm.index:
            row["Disclosed deals net ₹Cr"] = float(dm.loc[sec, "Total ₹Cr"])
            for c in CATEGORIES[:4]:
                row[f"Deals: {c}"] = float(dm.loc[sec, c])
        votes = (1 if flow21 > 0 else -1 if flow21 < 0 else 0) + (1 if len(acc) > len(dis) else -1 if len(dis) > len(acc) else 0)
        if row["FPI net ₹Cr (NSDL, fortnight)"] is not None:
            votes += 1 if row["FPI net ₹Cr (NSDL, fortnight)"] > 0 else -1
        if row["Disclosed deals net ₹Cr"] is not None:
            votes += 1 if row["Disclosed deals net ₹Cr"] > 0 else -1 if row["Disclosed deals net ₹Cr"] < 0 else 0
        row["Verdict"] = "ACCUMULATING" if votes >= 2 else "DISTRIBUTING" if votes <= -2 else "MIXED"
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Inferred flow 21d ₹Cr", ascending=False, na_position="last").reset_index(drop=True)


# ------------------------------------------------------------------------------
# One-stock summary of everything we know about "who, how much, for how long"
# ------------------------------------------------------------------------------
def hold_lines(symbol: str, fifo: pd.DataFrame | None, fp: dict | None, sh: pd.DataFrame | None, mf: pd.DataFrame | None) -> list[str]:
    lines = []
    if fifo is not None and len(fifo):
        f = fifo[fifo["Symbol"] == symbol]
        for _, r in f[f["State"] == "Holding"].sort_values("Open ₹Cr", ascending=False).head(5).iterrows():
            lines.append(f"**{r['Participant']}** ({r['Category']}) bought and has disclosed no matching sale: holding for at least "
                         f"**{r['Days held (min)']} days** (since {r['Open since']:%d %b %Y}), about ₹{r['Open ₹Cr']:,.1f} Cr still open.")
        for _, r in f[f["Avg closed hold (days)"].notna()].head(3).iterrows():
            lines.append(f"**{r['Participant']}** closed disclosed lots after an average **{r['Avg closed hold (days)']:.0f} days**.")
        for _, r in f[f["State"] == "Selling older holding"].head(3).iterrows():
            lines.append(f"**{r['Participant']}** is selling stock it bought before this window (₹{abs(r['Net ₹Cr']):,.1f} Cr net sold).")
    if sh is not None and len(sh):
        for _, r in sh[sh["Category"].isin(["FII/FPI", "DII (total)", "Mutual Fund", "Insurance"])].iterrows():
            if r["Streak (quarters)"]:
                lines.append(f"{r['Category']} have been **{r['Direction'].lower()} for {abs(int(r['Streak (quarters)']))} straight quarter(s)** "
                             f"(now {r['Latest %']:.2f}%, {r['QoQ change (pp)']:+.2f} pp QoQ) - i.e. a holding period of at least {abs(int(r['Streak (quarters)'])) * 3} months.")
    if mf is not None and len(mf):
        m = mf[mf["Stock"] == symbol]
        for _, r in m.sort_values("Value ₹Cr (latest)", ascending=False).head(5).iterrows():
            lines.append(f"**{r['Fund house']}**: {r['Action']}, ₹{r['Value ₹Cr (latest)']:,.1f} Cr held, in the portfolio for {r['Months held (streak)']} consecutive month(s).")
    if fp and fp.get("ok"):
        lines.append(f"Price-volume footprint (inferred): **{fp['phase']}** - {window_text(fp)}. "
                     f"Money-flow proxy: ₹{fp['flow5_cr']:+,.0f} Cr (5d), ₹{fp['flow21_cr']:+,.0f} Cr (21d), ₹{fp['flow63_cr']:+,.0f} Cr (63d).")
    return lines
