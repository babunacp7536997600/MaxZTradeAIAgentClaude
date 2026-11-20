"""
Offline integration test: the real data.scan_master() pipeline (indicators -> calibration -> footprint -> analysis) on synthetic
prices, then the Institutional radar roll-up and the AI Precision Agent brief.  No market connection is made.
Third-party packages that are not installed (streamlit / yfinance / feedparser) are replaced by tiny stubs.
"""
import os, sys, types, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd


def _stub(name, **attrs):
    try:
        __import__(name)
    except Exception:
        m = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(m, k, v)
        sys.modules[name] = m


def _cache_data(*a, **k):
    def deco(fn):
        fn.clear = lambda: None
        return fn
    return deco(a[0]) if a and callable(a[0]) else deco


_stub("streamlit", cache_data=_cache_data, session_state={}, secrets={})
_stub("yfinance", Ticker=lambda *a, **k: None)
_stub("feedparser", parse=lambda *a, **k: types.SimpleNamespace(entries=[]))

import agent as A          # noqa: E402
import data as D           # noqa: E402
import engine as E         # noqa: E402
import inst_engine as IE   # noqa: E402
from knowledge import BENCH, SECTOR_STOCKS, master_symbols  # noqa: E402
from synth import make_prices  # noqa: E402


class TestScanPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        end = pd.Timestamp.now().normalize()
        syms = master_symbols()[:40]
        frames = {}
        for i, s in enumerate(syms):
            df = make_prices(n=800, seed=i, drift=0.0005)
            df.index = pd.bdate_range(end=end, periods=len(df))
            frames[s + ".NS"] = df
        b = make_prices(n=800, seed=999)
        b.index = pd.bdate_range(end=end, periods=len(b))
        frames[BENCH] = b
        cls._orig = D._download_daily_cached
        D._download_daily_cached = lambda symbols, period="3y": frames
        cls.scan = D._scan_master_cached()

    @classmethod
    def tearDownClass(cls):
        D._download_daily_cached = cls._orig

    def test_scan_has_results_and_footprint(self):
        self.assertGreater(self.scan["n"], 30)
        self.assertEqual(set(self.scan["results"]), {k for k, v in self.scan["footprint"].items()})
        self.assertTrue(all("turnover_cr" in r for r in self.scan["results"].values()))
        self.assertTrue(any(v.get("ok") for v in self.scan["footprint"].values()))

    def test_sector_summary_from_scan(self):
        out = IE.sector_summary(self.scan["footprint"], IE.normalise_deals(None), SECTOR_STOCKS)
        self.assertGreater(len(out), 3)
        self.assertTrue(set(out["Verdict"]) <= {"ACCUMULATING", "DISTRIBUTING", "MIXED"})

    def test_inst_matrix_column_used_by_app(self):
        # app.py shows this column in the sector drill-down; the old name ("Calibrated tech prob") made it raise KeyError
        out = D.build_inst_matrix(self.scan, None, None)
        h = next(iter(out["stock"]))
        stk = out["stock"][h]
        self.assertGreater(len(stk), 0)
        for col in ["Stock", "Name", "Buy %", "Sell %", "Footprint", "CMF", "Vol x", "1D %", "Technical confluence index", "Status", "LTP", "Entry", "Target", "Stop"]:
            self.assertIn(col, stk.columns)

    def test_agent_brief_runs_on_scan(self):
        gate = A.Gate(horizon="5 Trading Days", min_turnover_cr=0.0, min_prob=50)
        for side in ("BUY", "SELL"):
            picks, watch = A.brief(self.scan, side, gate, {"RELIANCE": 5.0}, {"target_atr": 0.9, "stop_atr": 0.6})
            self.assertTrue(picks.empty or (picks["Side"] == side).all())


if __name__ == "__main__":
    unittest.main()
