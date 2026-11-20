"""Synthetic OHLCV helpers for offline tests (no market connection)."""
import numpy as np
import pandas as pd


def make_prices(n=1100, seed=0, drift=0.0004, vol=0.015, start="2021-01-01"):
    rng = np.random.default_rng(seed)
    ret = rng.normal(drift, vol, n)
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]] * (1 + rng.normal(0, 0.003, n))
    span = np.abs(rng.normal(0, vol, n)) * close
    high = np.maximum(open_, close) + span * 0.6
    low = np.minimum(open_, close) - span * 0.6
    volu = rng.integers(500_000, 3_000_000, n).astype(float) * (1 + 0.8 * (ret > 0.02))
    idx = pd.bdate_range(start, periods=n)
    return pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volu}, index=idx)
