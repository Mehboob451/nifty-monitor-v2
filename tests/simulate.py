"""Simulates full trading days with synthetic data so the whole pipeline (indicators, options, engine,
alerts, scoreboard) can be tested without internet. Usage: python tests/simulate.py [days] [out_dir]"""
import datetime as dt
import os
import shutil
import sys

import numpy as np
import pandas as pd

OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/nifty_sim"
os.environ["DATA_DIR"] = OUT
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import main as M                       # noqa: E402
from src.config import GLOBAL_CUES, IST         # noqa: E402
from src.util import clip                       # noqa: E402

rng = np.random.default_rng(7)
FAIL = os.environ.get("DEMO_FAIL", "")


def session(date, start_price, vol=0.00011, drift=0.0):
    idx = pd.date_range(dt.datetime.combine(date, dt.time(9, 15)), periods=375, freq="1min", tz=IST)
    r = rng.normal(drift, vol, len(idx))
    close = start_price * np.exp(np.cumsum(r))
    open_ = np.concatenate([[start_price], close[:-1]])
    hi = np.maximum(open_, close) * (1 + np.abs(rng.normal(0, 0.00004, len(idx))))
    lo = np.minimum(open_, close) * (1 - np.abs(rng.normal(0, 0.00004, len(idx))))
    return pd.DataFrame({"Open": open_, "High": hi, "Low": lo, "Close": close, "Volume": 0}, index=idx)


class Demo:
    def __init__(self, dates):
        px, frames = 25000.0, []
        for i, d in enumerate(dates):
            df = session(d, px, drift=float(os.environ.get("DEMO_DRIFT", "0.00002")) * rng.normal(0, 1))
            frames.append(df)
            px = float(df.Close.iloc[-1])
        self.full = pd.concat(frames)
        dd = self.full.groupby(self.full.index.date).agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
        dd.index = pd.DatetimeIndex([pd.Timestamp(x, tz=IST) for x in dd.index])
        self.daily = dd
        self.calls = 0
        self.oi = None

    def nifty(self, now):
        m = self.full[self.full.index <= pd.Timestamp(now) - pd.Timedelta(minutes=1)]
        d = self.daily[self.daily.index.date < now.date()]
        return m, d

    def global_(self, today):
        if "global" in FAIL:
            return [], None
        out, num, den = [], 0, 0
        for sym, (label, sgn, w, scale) in GLOBAL_CUES.items():
            pct = float(rng.normal(0, 0.6))
            sig = clip(sgn * pct / scale)
            num += w * sig
            den += w
            out.append({"symbol": sym, "label": label, "price": (14 + pct if sym == "^INDIAVIX" else 100 + pct), "change_pct": pct, "signal": sig, "stale": False, "weight": w})
        return out, clip(num / den)

    def breadth(self, today, nifty_pct):
        from src.config import HEAVYWEIGHTS
        items = [{"symbol": k, "label": v[0], "change_pct": float(rng.normal(0, 0.7)), "weight": v[1]} for k, v in HEAVYWEIGHTS.items()]
        hv = sum(i["weight"] * i["change_pct"] for i in items) / sum(i["weight"] for i in items)
        return items, clip(hv / 0.6), {"heavy_pct": hv, "bank_pct": 0.1, "lead": hv,
                                        "parts": [{"name": "Heavyweights + Bank Nifty", "value": clip(hv / 0.6), "weight": 0.65, "note": "demo"}], "note": None}

    def chain(self):
        if "chain" in FAIL:
            return None, ["HTTP 403"]
        self.calls += 1
        m = self.full[self.full.index <= pd.Timestamp(self.now) - pd.Timedelta(minutes=1)]
        spot = float(m.Close.iloc[-1])
        atm = round(spot / 50) * 50
        if self.oi is None:
            ks = np.arange(atm - 2000, atm + 2001, 50)
            self.oi = {float(k): [1.2e6 * np.exp(-((k - (atm + 350)) / 500) ** 2) + 3e4, 1.2e6 * np.exp(-((k - (atm - 350)) / 500) ** 2) + 3e4] for k in ks}
            self.base = {k: list(v) for k, v in self.oi.items()}
        for k, v in self.oi.items():
            v[0] *= 1 + rng.normal(0.002, 0.012)
            v[1] *= 1 + rng.normal(0.002, 0.012)
        if self.calls == 9:            # inject a "big fish" put-writing spike below spot
            self.oi[float(atm - 100)][1] *= 2.4
        rows = [{"strike": k, "ce_oi": v[0], "ce_chg": v[0] - self.base[k][0], "ce_vol": 0, "ce_iv": 13.0, "ce_ltp": 10,
                 "pe_oi": v[1], "pe_chg": v[1] - self.base[k][1], "pe_vol": 0, "pe_iv": 13.5, "pe_ltp": 10} for k, v in self.oi.items()]
        expiry = (self.now.date() + dt.timedelta(days=3)).strftime("%d-%b-%Y")
        return {"expiry": expiry, "spot": spot, "timestamp": "", "rows": rows}, []

    def news(self):
        from src.news import keyword_score  # noqa: F401
        return [{"title": "Sensex rallies as FII buying returns", "source": "Demo", "time": self.now, "link": ""},
                {"title": "Crude oil slides on ceasefire hopes", "source": "Demo", "time": self.now, "link": ""}]


if __name__ == "__main__":
    ndays = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    shutil.rmtree(OUT, ignore_errors=True)
    # weekdays ending on Friday 2026-09-18 (+ 5 warm-up sessions before the simulated days)
    all_days = pd.bdate_range(end="2026-09-18", periods=ndays + 5).date
    demo = Demo(all_days)
    for day in all_days[-ndays:]:
        demo.calls, demo.oi = 0, None
        t = dt.datetime.combine(day, dt.time(9, 25), tzinfo=IST)
        end = dt.datetime.combine(day, dt.time(16, 5), tzinfo=IST)
        while t <= end:
            demo.now = t
            M.run(t, demo)
            t += dt.timedelta(minutes=15)
    print("\nWrote:", os.listdir(OUT))
