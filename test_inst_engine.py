import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import engine as E
import inst_engine as IE
from synth import make_prices


def deals_frame():
    rows = [
        ["10-Nov-2026", "RELIANCE", "MORGAN STANLEY ASIA (SINGAPORE) PTE.", "BUY", "1,000,000", "1,500"],
        ["20-Nov-2026", "RELIANCE", "MORGAN STANLEY ASIA (SINGAPORE) PTE.", "SELL", "400,000", "1,550"],
        ["12-Nov-2026", "HDFCBANK", "SBI MUTUAL FUND - SBI CONTRA FUND", "BUY", "2,000,000", "1,000"],
        ["15-Nov-2026", "TCS", "Life Insurance Corporation of India", "SELL", "500,000", "4,000"],
        ["15-Nov-2026", "TCS", "Modi Securities Pvt Ltd", "BUY", "10,000", "4,000"],
    ]
    return pd.DataFrame(rows, columns=["Date", "Symbol", "Client Name", "Buy/Sell", "Quantity Traded", "Trade Price / Wght. Avg. Price"])


class TestClassification(unittest.TestCase):
    def test_categories(self):
        self.assertEqual(IE.classify_client("MORGAN STANLEY ASIA (SINGAPORE) PTE."), "FII/FPI")
        self.assertEqual(IE.classify_client("SBI MUTUAL FUND - SBI CONTRA FUND"), "Mutual Fund")
        self.assertEqual(IE.classify_client("NIPPON LIFE INDIA TRUSTEE LTD- A/C NIPPON INDIA GROWTH FUND"), "Mutual Fund")
        self.assertEqual(IE.classify_client("Life Insurance Corporation of India"), "Insurance")
        self.assertEqual(IE.classify_client("Modi Securities Pvt Ltd"), "Prop / HNI / Other")  # 'odi' must not match inside 'Modi'
        self.assertEqual(IE.classify_client("Magic Capital"), "Prop / HNI / Other")           # 'gic' must not match inside 'Magic'
        self.assertEqual(IE.classify_client("Employees Provident Fund Organisation"), "Domestic Institution")

    def test_amc(self):
        self.assertEqual(IE.amc_of("SBI MUTUAL FUND - SBI CONTRA FUND"), "SBI Mutual Fund")
        self.assertEqual(IE.amc_of("NIPPON LIFE INDIA TRUSTEE LTD- A/C NIPPON INDIA GROWTH FUND"), "Nippon Life India")


class TestDeals(unittest.TestCase):
    def setUp(self):
        self.d = IE.normalise_deals(deals_frame())

    def test_normalise(self):
        self.assertEqual(len(self.d), 5)
        r = self.d[(self.d.Symbol == "RELIANCE") & (self.d.Side == "BUY")].iloc[0]
        self.assertAlmostEqual(r.Value_Cr, 1_000_000 * 1500 / 1e7)
        self.assertEqual(r.Category, "FII/FPI")
        self.assertEqual(r.Sector, "Oil, Gas & Energy")

    def test_bad_input(self):
        self.assertTrue(IE.normalise_deals(None).empty)
        self.assertTrue(IE.normalise_deals(pd.DataFrame({"a": [1]})).empty)

    def test_board_and_matrix(self):
        b = IE.participant_board(self.d)
        ms = b[b.Participant.str.startswith("Morgan")].iloc[0]
        self.assertAlmostEqual(ms["Net ₹Cr"], 150.0 - 62.0)
        self.assertIn("RELIANCE", ms["Stocks being bought (₹Cr)"])
        sm = IE.sector_matrix(self.d)
        self.assertEqual(set(IE.CATEGORIES) - set(sm.columns), set())
        self.assertAlmostEqual(float(sm[sm.Sector == "Financial Services"]["Mutual Fund"].iloc[0]), 200.0)

    def test_fifo(self):
        f = IE.fifo_positions(self.d, asof="2026-11-30")
        r = f[(f.Symbol == "RELIANCE")].iloc[0]
        self.assertEqual(r["State"], "Holding")
        self.assertEqual(r["Open qty"], 600000)
        self.assertEqual(r["Days held (min)"], 20)
        self.assertEqual(r["Avg closed hold (days)"], 10.0)
        t = f[(f.Symbol == "TCS") & (f.Participant.str.contains("Life"))].iloc[0]
        self.assertEqual(t["State"], "Selling older holding")


class TestFootprint(unittest.TestCase):
    def test_runs(self):
        m = np.array([0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 1, 1], bool)
        self.assertEqual(IE._runs(m, max_gap=1, min_len=3), [(1, 6), (11, 15)])
        self.assertEqual(IE._runs(m, max_gap=0, min_len=3), [(1, 3), (11, 15)])

    def test_stats(self):
        fp = IE.footprint_stats(E.ind(make_prices(seed=3)))
        self.assertTrue(fp["ok"])
        self.assertIn(fp["phase"], ("Accumulation", "Distribution", "Neutral"))
        self.assertIsInstance(IE.window_text(fp), str)
        self.assertFalse(IE.footprint_stats(None)["ok"])

    def test_sector_summary(self):
        syms = ["AAA", "BBB"]
        foot = {s: IE.footprint_stats(E.ind(make_prices(seed=i))) for i, s in enumerate(syms)}
        out = IE.sector_summary(foot, IE.normalise_deals(None), {"Test": syms})
        self.assertEqual(len(out), 1)
        self.assertIn(out.iloc[0]["Verdict"], ("ACCUMULATING", "DISTRIBUTING", "MIXED"))


class TestShareholding(unittest.TestCase):
    def test_wide_and_changes(self):
        raw = pd.DataFrame({"Category": ["Promoters", "FIIs", "DIIs", "Public"],
                            "Mar 2025": [50.0, 20.0, 15.0, 15.0], "Jun 2025": [50.0, 21.0, 15.0, 14.0],
                            "Sep 2025": [50.0, 22.5, 14.5, 13.0], "Dec 2025": [50.0, 23.0, 14.0, 13.0]})
        long = IE.parse_shareholding(raw)
        ch = IE.shareholding_changes(long, market_cap_cr=100000).set_index("Category")
        self.assertEqual(ch.loc["FII/FPI", "Streak (quarters)"], 3)
        self.assertEqual(ch.loc["FII/FPI", "Direction"], "Adding")
        self.assertAlmostEqual(ch.loc["FII/FPI", "Approx ₹Cr moved (QoQ)"], 500.0)
        self.assertEqual(ch.loc["DII (total)", "Streak (quarters)"], -2)

    def test_rows_are_quarters(self):
        raw = pd.DataFrame({"Quarter": ["2025-03-31", "2025-06-30", "2025-09-30"], "FII": [10, 11, 12], "DII": [20, 19, 18]})
        long = IE.parse_shareholding(raw)
        self.assertEqual(long["Quarter"].nunique(), 3)
        self.assertEqual(set(long["Category"]), {"FII/FPI", "DII (total)"})


class TestAmcPortfolio(unittest.TestCase):
    def sheet(self, qty_pre, qty_cur):
        title = [["Monthly Portfolio Statement of XYZ Fund"], ["(Rs. in Lakhs)"], [""]]
        head = [["Name of the Instrument", "ISIN", "Industry", "Quantity", "Market/Fair Value (Rs. in Lakhs)", "% to NAV"]]
        rows = [["Reliance Industries Limited", "INE002A01018", "Petroleum", qty_cur, "250000", "5.1"],
                ["HDFC Bank Limited", "INE040A01034", "Banks", "100000", "160000", "4.0"],
                ["Some Bond 8%", "INE123X07015", "Debt", "1", "1000", "0.1"]]
        return pd.DataFrame(title + head + rows)

    def test_parse_and_changes(self):
        a = IE.parse_amc_portfolio(self.sheet(0, "1000000"), "XYZ Mutual Fund", "2026-09-01")
        b = IE.parse_amc_portfolio(self.sheet(0, "1200000"), "XYZ Mutual Fund", "2026-10-01")
        self.assertEqual(len(a), 2)                                   # debt line dropped
        self.assertEqual(set(a["Symbol"]), {"RELIANCE", "HDFCBANK"})
        self.assertAlmostEqual(float(a[a.Symbol == "RELIANCE"]["Value_Cr"].iloc[0]), 2500.0)   # lakhs -> crore
        ch = IE.mf_changes(pd.concat([a, b])).set_index("Stock")
        self.assertEqual(ch.loc["RELIANCE", "Action"], "ADDED")
        self.assertEqual(ch.loc["RELIANCE", "Months held (streak)"], 2)
        self.assertAlmostEqual(ch.loc["RELIANCE", "Net ₹Cr bought / (sold)"], 200000 * 2500.0 / 1_200_000, places=1)
        self.assertEqual(ch.loc["HDFCBANK", "Action"], "UNCHANGED")

    def test_no_header(self):
        self.assertTrue(IE.parse_amc_portfolio(pd.DataFrame([["a", "b"]]), "X", "2026-01-01").empty)


if __name__ == "__main__":
    unittest.main()
