"""Backtest on synthetic candles: random walk should adjust nothing; a momentum market should find an edge."""
import datetime as dt, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import backtest
from src.config import IST

def make(phi, days=60, seed=3):
    rng = np.random.default_rng(seed)
    frames, px, r_prev = [], 25000.0, 0.0
    dates = pd.bdate_range(end="2026-09-18", periods=days).date
    for d in dates:
        idx = pd.date_range(dt.datetime.combine(d, dt.time(9, 15)), periods=75, freq="5min", tz=IST)
        r = np.zeros(75)
        for k in range(75):
            r[k] = phi * r_prev + rng.normal(0, 0.0006); r_prev = r[k]
        close = px * np.exp(np.cumsum(r)); open_ = np.concatenate([[px], close[:-1]])
        frames.append(pd.DataFrame({"Open": open_, "High": np.maximum(open_, close) * 1.0002, "Low": np.minimum(open_, close) * 0.9998, "Close": close, "Volume": 0}, index=idx))
        px = close[-1]
    m5 = pd.concat(frames)
    dd = m5.groupby(m5.index.date).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    dd.index = pd.DatetimeIndex([pd.Timestamp(x, tz=IST) for x in dd.index])
    return m5, dd

for phi in (0.0, 0.25):
    m5, dd = make(phi)
    rep, mult = backtest.run(m5, dd)
    print(f"phi={phi}: samples={rep['samples']} days={rep['days']} adjusted={mult}")
    for r in rep["indicators"][:4]:
        print("   ", r["name"], "n", r["n30"], "hit30", None if r["hit30"] is None else round(r["hit30"], 3), "z", round(r["z30"], 1))
    print("    total score >=0.25:", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in rep["total_score"][2].items() if k in ("n30", "hit30", "z30")})
