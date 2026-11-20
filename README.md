# Trader Dashboard Pro — Advanced Research, Institutional Radar & AI Precision Agent Edition

A Streamlit research terminal for Indian equities, indices and mutual funds. This edition preserves the original dashboard modules and adds a **Performance Lab** for chronological out-of-sample testing, an **🏛️ Institutional Holdings Radar** (who is buying / selling what, how much, in which sector and stock, and for how long) and an **🧠 AI Precision Agent** (a multi-check gate whose accuracy is *measured on held-out trades across a whole stock universe*).

> **No software can guarantee an 80–90% win rate.** The dashboard's existing model-confidence index is a ranking/confluence score, not a calibrated promise of future win probability. Use the Performance Lab to inspect measured results and uncertainty before risking capital.

## What's included

| Page | Features |
|---|---|
| Market & FII Radar | FII/DII flows, sector/stock radar, NSDL sector flows, inferred price-volume footprint, sector momentum |
| Watchlist & Pinned | Track stocks and funds, signal levels, backup and restore |
| Search & Multi-TF Engine | Stock, index and mutual fund analysis; indicators, ATR plans, multi-timeframe candles, news and price/volume zones |
| Stocks (High Conviction) | NSE stock ideas ranked by model confidence index across multiple horizons |
| Mutual Funds (Ranked) | NAV-based fund ranking and technical analysis |
| AI Quant Copilot | English/Hinglish research helper; optional Gemini key |
| News & Forex Factory | Macro calendar and headline classification |
| **🏛️ Institutional Holdings (new)** | Named FII/FPI, MF houses, insurers, DIIs and other big players: net ₹ Cr bought / sold, sectors and stocks, observed minimum holding period, typical accumulation-phase length, fund-house month-on-month changes, shareholding-pattern streaks |
| **🧠 AI Precision Agent (new)** | Gate with an explainable checklist per stock, live brief of setups that pass every check, universe-wide walk-forward verification with train-only parameter selection, goal tracker that shows ✅ only when held-out trades prove the target |
| **Performance Lab** | Core vs strict confluence comparison, chronological held-out results, win-rate confidence interval, target-first rate, profit factor, expectancy, drawdown, equity curve and CSV ledger |

## 🏛️ Institutional Holdings Radar

Answers: *which FII / FPI / mutual-fund house / insurer / DII / other big player is putting in or pulling out how much money, in which sector and stock, and for how many days will they hold?*

| Question | What the app uses | Real or inferred |
|---|---|---|
| Who, how much, which stock | NSE bulk + block deals (named client, side, quantity, price) | **Real, disclosed** (large trades only) |
| Which fund house holds / adds / cuts what | AMC monthly portfolio files you upload (equity lines are detected automatically, Lakhs → Crore) | **Real, disclosed** (monthly) |
| FII / DII / MF / insurer % and trend | Shareholding-pattern file you upload per stock (quarters as rows or columns) | **Real, disclosed** (quarterly) |
| Sector money map | NSDL FPI fortnightly sector flow + deal net by participant type + price-volume money-flow proxy | Mixed, each column labelled |
| How long will they hold | (a) *observed minimum hold*: FIFO-matched bulk/block lots, shareholding streaks, months in an AMC portfolio; (b) *typical length of the same stock's past accumulation / distribution phases* compared with the age of the current phase | **Estimate** - holding periods are never published in advance |

Notes
- Participants are classed (FII/FPI, Mutual Fund, Insurance, Domestic Institution, Prop/HNI/Other) from the client name in the exchange disclosure; unusual names can be misclassified.
- Bulk/block deals show only trades above exchange thresholds. A participant may also buy or sell smaller quantities that are not visible.
- NSE often blocks cloud servers. If the live fetch returns nothing the page says so and you can upload the CSV downloaded from NSE (Reports → Bulk deals / Block deals). No illustrative numbers are shown in this page.
- Yahoo Finance holder tables are shown as a best-effort extra; coverage of Indian stocks is patchy.

## 🧠 AI Precision Agent

> **No software can guarantee an 80–90% win rate.** The agent's job is to make accuracy *measurable*: it trades rarely, explains every decision and tells you the held-out hit-rate of its own gate.

1. **Gate** - a stock is an agent pick only if it passes *every* check: signal direction and strength, strict EMA/MACD/RSI/CMF/volume confluence, Nifty regime, price-volume footprint (accumulation for BUY, distribution for SELL), not stretched from EMA21, phase not exhausted, liquidity, expected move ≥ 5× round-trip cost, and disclosed deals not against the trade. Checks that were back-tested and live-only overlays are labelled. Stocks that fail exactly one check appear as near-misses.
2. **Verification** - runs the chronological walk-forward test (next-session Open entry, stop-first on ambiguous candles, costs deducted) over a whole universe so the result rests on hundreds of trades. Settings (gate, score, target ATR, stop ATR) are **chosen on the training period only** and the later held-out trades of that one configuration are reported next to the full train-vs-test table.
3. **Goal tracker** - shows *VERIFIED ≥ 80%* only when held-out trades satisfy: ≥ 100 trades, win rate ≥ 80%, 95% interval lower bound ≥ 70%, positive expectancy after costs and profit factor ≥ 1.3. Otherwise it shows the measured number and which criterion failed.
4. A tiny target with a wide stop produces a high win rate and can still lose money, so selection always requires positive expectancy and profit factor > 1 on the training trades.
5. The live brief uses the verified target / stop multipliers (√time × ATR) and prints the measured held-out hit-rate beside the picks.

The Copilot also understands *"agent picks swing buy"*, *"kaun si fund house kya buy kar rahi hai"* and *"RELIANCE me FII kitne din hold karenge"*.

## Performance Lab methodology

The lab supports 1-day daily-bar proxy and 2/3/4/5-day short swing windows as well as 2-week, 2–5-week, 1-month, 3-month and 6-month+ horizons. It deliberately uses stricter assumptions than a simple close-to-close hit-rate:

1. Indicators and the signal are calculated after the daily candle closes.
2. The paper trade enters at the **next trading session's Open**. It does not assume a fill at the signal close.
3. ATR is taken from the signal candle, before the trade begins.
4. Target and stop levels are fixed from the entry price using configurable ATR multipliers and the selected horizon.
5. Opening gaps are handled at the observed Open. If target and stop are both inside the same daily candle, the backtest assumes the **stop happened first**, because daily OHLC does not reveal the intraday path.
6. A user-specified round-trip cost/slippage estimate is subtracted from every trade.
7. Only a chronological test segment after the initial history split is scored. A single position per symbol is allowed, avoiding overlapping trades.
8. A 95% Wilson interval is displayed with the win rate. A direction-specific probability estimate is learned only from earlier training-period completed trades, then evaluated on later held-out trades with a Brier score. Low trade counts and wide intervals are flagged.

**Core rule vs strict filter:** The strict filter additionally requires aligned EMA9/EMA21 structure, price vs EMA21, positive/negative MACD histogram, directional RSI range and CMF confirmation. It may reduce weak setups, but it may also reduce the number of trades or perform worse. The comparison table shows both results instead of claiming the strict filter must be better.

### Important limitations

- The stock data source is Yahoo Finance through `yfinance`; it can be delayed, throttled, corrected or unavailable. It is not an exchange-grade tick feed.
- The test uses currently available historical constituents and does not remove survivorship bias.
- Corporate actions, circuit limits, order queue position, market impact, borrow availability, taxes by individual trade and broker-specific charges are not fully modelled. Enter a realistic round-trip cost estimate.
- Daily candles cannot identify whether the high or low occurred first. The 1-Day horizon in this lab is still a daily-bar proxy, not a true intraday 1-minute/5-minute simulation. The stop-first assumption is conservative, but it is not a substitute for intraday data.
- The SELL side is a hypothetical bearish direction test. Multi-day cash-market short selling is not automatically available; verify instrument, product and exchange rules.
- A model can overfit when thresholds are repeatedly changed after looking at the test results. Freeze your rule first, then use genuinely later paper-trading data as a further validation set.
- Even a high observed win rate can lose money if average losses exceed average wins. Review net expectancy, profit factor, drawdown and sample size together.

## Run locally on Windows

1. Install Python 3.10–3.13 from [python.org](https://www.python.org/downloads/) and select **Add Python to PATH**.
2. Open this project folder in Command Prompt.
3. Install packages:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Start the app:

   ```bash
   streamlit run app.py
   ```

   Or double-click `run_app.bat`.

Open `http://localhost:8501` in your browser. The first scan may take time because historical prices are downloaded. Use **Refresh Terminal** to refresh cached data.

## Deploy on GitHub + Streamlit Community Cloud

The ZIP contains `app.py` at the project root, `requirements.txt`, `.streamlit/config.toml`, `.gitignore` and a GitHub Actions workflow.

1. Extract the ZIP.
2. Create a new GitHub repository.
3. Upload the **contents** of the `trader_dashboard` folder to the repository root - including the `.github` and `.streamlit` folders and `.gitignore` (not the enclosing ZIP and not a nested folder).
4. Commit and push to the `main` branch.
5. Open [Streamlit Community Cloud](https://share.streamlit.io/), choose **New app**, select your repository, branch `main`, and main file path `app.py`, then deploy.
6. The app installs dependencies from `requirements.txt` automatically.

### Optional Gemini key

The AI Copilot can run without a Gemini key using its built-in rule-based helper. To enable the optional Gemini integration, add this under **App → Settings → Secrets** on Streamlit Cloud:

```toml
GEMINI_API_KEY = "your-key"
```

Do not commit API keys or `.streamlit/secrets.toml` to GitHub.

## Run tests

Offline unit tests (no market connection required):

```bash
python -m unittest discover -s tests -v
```


The GitHub Actions workflow (`.github/workflows/ci.yml`) compiles every module and runs the offline unit and integration tests on each push / pull request. `tests/test_integration.py` runs the real scan pipeline (indicators → calibration → footprint → agent) on synthetic prices, so no market connection is needed.

## Data and privacy notes

- Prices, volume and candles: Yahoo Finance. Mutual-fund NAV: MFAPI/AMFI source.
- FII/DII, NSDL and bulk/block deal pages may be unavailable from cloud hosts. The app labels unavailable or illustrative sources rather than presenting them as verified live disclosures.
- Stock-level institutional entry prices are not published as a complete daily feed. Price-volume “big-player” zones are inferred, not actual FII/FPI account positions.
- Streamlit Community Cloud storage is ephemeral. Use Watchlist → Backup / Restore to preserve your watchlist.
- Files you upload on the Institutional Holdings page (deals, AMC portfolios, shareholding) live only in your browser session and are not stored by the app.

## Disclaimer

This is an algorithmic research and educational tool, not investment advice and not a SEBI-registered advisory service. Historical or simulated performance is not a guarantee of future returns. Verify data, costs, liquidity, circuit limits, corporate actions and order execution before any live trade. Use position sizing and predefined risk limits.
