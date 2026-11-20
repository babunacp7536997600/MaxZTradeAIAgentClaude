import os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd

import agent as A
import engine as E
import inst_engine as IE
import precision_lab as P
import validation as V
from synth import make_prices


class TestValidationExtensions(unittest.TestCase):
    def setUp(self):
        self.px = make_prices(seed=7)
        self.reg, self.bench = E.market_regime(make_prices(seed=99))

    def test_prepared_matches_unprepared(self):
        kw = dict(horizon_label="5 Trading Days", threshold=40, strict_filter=True, regime=self.reg, bench=self.bench)
        a = V.walk_forward_backtest(self.px, **kw)
        b = V.walk_forward_backtest(self.px, prepared=V.prepare(self.px, self.reg, self.bench), **kw)
        self.assertEqual(a["metrics"]["trades"], b["metrics"]["trades"])
        pd.testing.assert_frame_equal(a["trades"].reset_index(drop=True), b["trades"].reset_index(drop=True))

    def test_train_trades_end_before_test_and_precision_is_subset(self):
        r = V.walk_forward_backtest(self.px, horizon_label="5 Trading Days", threshold=30, regime=self.reg, bench=self.bench)
        split = r["metadata"]["split_date"]
        self.assertTrue((r["train_trades"]["exit_date"] < split).all())
        self.assertTrue((r["trades"]["entry_date"] >= split).all())
        core = V.walk_forward_backtest(self.px, horizon_label="5 Trading Days", threshold=30, regime=self.reg, bench=self.bench)
        prec = V.walk_forward_backtest(self.px, horizon_label="5 Trading Days", threshold=30, regime=self.reg, bench=self.bench, precision_filter=True)
        self.assertLessEqual(prec["metadata"]["test_candidate_signals"], core["metadata"]["test_candidate_signals"])

    def test_next_open_entry_and_costs(self):
        r = V.walk_forward_backtest(self.px, horizon_label="5 Trading Days", threshold=30, cost_bps=50, regime=self.reg, bench=self.bench)
        t = r["trades"].iloc[0]
        self.assertGreater(t["entry_date"], t["signal_date"])
        self.assertAlmostEqual(t["entry"], float(self.px.loc[t["entry_date"], "Open"]))
        self.assertAlmostEqual(t["gross_return_pct"] - t["net_return_pct"], 0.5, places=6)


class TestPrecisionLab(unittest.TestCase):
    def test_pooled_metrics_and_goal(self):
        n = 120
        tr = pd.DataFrame({"net_return_pct": [1.0] * 100 + [-1.0] * 20, "r_multiple": [1.0] * 100 + [-1.0] * 20, "target_first": [True] * 100 + [False] * 20,
                           "side": ["BUY"] * n, "symbol": ["A"] * 60 + ["B"] * 60, "exit_date": pd.date_range("2025-01-01", periods=n)})
        m = P.pooled_metrics(tr)
        self.assertAlmostEqual(m["win_rate"], 100 / 120 * 100)
        self.assertAlmostEqual(m["profit_factor"], 5.0)
        self.assertEqual(m["symbols"], 2)
        g = P.goal_check(m, 80.0)
        self.assertTrue(g["verified"])                         # 120 trades, 83.3%, Wilson low ~76%, PF 5
        small = tr.head(30).copy()
        small["net_return_pct"] = [1.0] * 25 + [-1.0] * 5      # 83% but only 30 trades -> must NOT be verified
        small["r_multiple"] = small["net_return_pct"]
        gs = P.goal_check(P.pooled_metrics(small), 80.0)
        self.assertFalse(gs["verified"])
        self.assertFalse(dict((c[0], c[1]) for c in gs["checks"])["At least 100 held-out trades"])
        weak = tr.copy()
        weak["net_return_pct"] = [1.0] * 70 + [-1.0] * 50
        weak["r_multiple"] = weak["net_return_pct"]
        self.assertFalse(P.goal_check(P.pooled_metrics(weak), 80.0)["verified"])
        self.assertFalse(P.goal_check(P.pooled_metrics(pd.DataFrame()), 80.0)["verified"])

    def test_optimizer_selects_on_train_only(self):
        prices = {f"S{i}": make_prices(seed=i) for i in range(6)}
        reg, bench = E.market_regime(make_prices(seed=99))
        grid = {"gate": ["strict"], "threshold": [40], "target_atr": [0.6, 1.2], "stop_atr": [0.6, 1.0]}
        res = P.optimize(prices, reg, bench, horizon="5 Trading Days", grid=grid, min_train_trades=20)
        self.assertEqual(len(res["table"]), 4)
        self.assertIn("reason", res)
        if res["selected_key"]:
            row = res["table"][(res["table"]["Target ATR"] == res["selected_key"][2]) & (res["table"]["Stop ATR"] == res["selected_key"][3])].iloc[0]
            el = res["table"][res["table"]["Eligible"]]
            self.assertEqual(row["Train win %"], el["Train win %"].max())      # chosen by TRAIN win rate, not test
            self.assertTrue(row["Train expectancy R"] > 0)

    def test_random_walk_gets_no_edge(self):
        """Sanity: on edge-less random data the held-out win rate must not look like 80%."""
        prices = {f"S{i}": make_prices(seed=100 + i, drift=0.0) for i in range(8)}
        reg, bench = E.market_regime(make_prices(seed=5, drift=0.0))
        prep = {s: V.prepare(d, reg, bench) for s, d in prices.items()}
        r = P.run_universe(prices, prep, reg, bench, horizon="5 Trading Days", threshold=40, gate="strict")
        m = r["test_metrics"]
        self.assertGreater(m["trades"], 100)
        self.assertLess(m["win_rate"], 65.0)


class TestAgentGate(unittest.TestCase):
    def setUp(self):
        reg, bench = E.market_regime(make_prices(seed=99))
        self.scan = {"results": {}, "footprint": {}}
        for i in range(10):
            df = make_prices(seed=20 + i, drift=0.0006)
            res = E.analyze(df, None, reg, bench.reindex(df.index, method="ffill"), f"Stock {i}")
            res["sector"] = "Test"
            sym = f"T{i}"
            self.scan["results"][sym] = res
            self.scan["footprint"][sym] = IE.footprint_stats(res["_d"])

    def test_checklist_and_brief(self):
        gate = A.Gate(horizon="5 Trading Days", min_turnover_cr=0.0)
        res = next(iter(self.scan["results"].values()))
        ch = A.evaluate(res, "BUY", gate, self.scan["footprint"]["T0"])
        self.assertTrue(all({"check", "ok", "detail", "tested"} <= set(c) for c in ch))
        picks, watch = A.brief(self.scan, "BUY", gate)
        self.assertTrue(picks.empty or (picks["Failed check"] == "-").all())
        self.assertTrue(watch.empty or (watch["Failed check"] != "-").all())
        if not picks.empty:
            self.assertTrue(all(all(c["ok"] for c in row) for row in picks["_checks"]))

    def test_verified_plan_levels(self):
        res = next(iter(self.scan["results"].values()))
        gate = A.Gate(horizon="5 Trading Days")
        lv = A.levels(res, "BUY", gate, {"target_atr": 0.9, "stop_atr": 0.6})
        unit = np.sqrt(5) * res["atr"]
        self.assertAlmostEqual(lv["target"] - lv["entry"], 0.9 * unit, delta=0.02)
        self.assertAlmostEqual(lv["entry"] - lv["stop"], 0.6 * unit, delta=0.02)
        sell = A.levels(res, "SELL", gate, {"target_atr": 0.9, "stop_atr": 0.6})
        self.assertLess(sell["target"], sell["entry"])
        self.assertGreater(sell["stop"], sell["entry"])


if __name__ == "__main__":
    unittest.main()
