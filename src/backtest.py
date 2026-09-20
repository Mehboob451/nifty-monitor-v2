"""Backtest of the technical indicators on ~60 days of 5-minute Nifty candles (free Yahoo data).

For every decision point (every 15 minutes, 9:45 to 14:30 IST) it computes each indicator using ONLY data
available at that moment, then checks whether its lean matched the move over the next 15/30/60 minutes.
Weights are only changed when an indicator shows a statistically strong edge that also holds on the
held-out last 30% of days, and never by more than +/-50%.

Run:  python -m src.backtest
"""
import datetime as dt
import math
import os

import numpy as np
import pandas as pd

from . import indicators
from .config import DATA_DIR, IST
from .util import ist_now, log, save_json

HORIZONS = (15, 30, 60)
MIN_N = 300            # min samples for an indicator to be judged
MIN_Z = 2.5            # strong evidence needed (many indicators are tested, so this is strict)
MAX_TWEAK = 0.5


def fetch():
    import yfinance as yf
    t = yf.Ticker("^NSEI")
    m5 = t.history(period="60d", interval="5m", auto_adjust=False)
    daily = t.history(period="1y", interval="1d", auto_adjust=False)
    if m5 is None or m5.empty or daily is None or daily.empty:
        return None, None
    m5.index = m5.index.tz_convert(IST) if m5.index.tz else m5.index.tz_localize(IST)
    daily.index = (daily.index.tz_convert(IST) if daily.index.tz else daily.index.tz_localize(IST)).normalize()
    cols = ["Open", "High", "Low", "Close", "Volume"]
    return m5[cols], daily[cols]


def collect(m5, daily):
    """One row per decision point: indicator values + realised forward moves."""
    idx = m5.index
    pos = {ts: i for i, ts in enumerate(idx)}
    close = m5["Close"].values
    rows = []
    for i in range(60, len(idx)):
        end = idx[i] + pd.Timedelta(minutes=5)
        hm = end.hour * 60 + end.minute
        if not (9 * 60 + 45 <= hm <= 14 * 60 + 30) or end.minute % 15 != 0:
            continue
        fwd, ok = {}, True
        for h in HORIZONS:
            j = pos.get(idx[i] + pd.Timedelta(minutes=h))
            if j is None or idx[j].date() != idx[i].date():
                ok = False
                break
            fwd[h] = close[j] / close[i] - 1
        if not ok:
            continue
        try:
            _, parts, score = indicators.compute(m5.iloc[max(0, i - 374):i + 1], daily, idx[i].date())
        except Exception:  # noqa: BLE001
            continue
        rows.append({"date": idx[i].date(), "score": score, "parts": {p["name"]: p["value"] for p in parts}, "fwd": fwd})
    return rows


def stats(sig, fwd):
    """Hit rate of sign(sig) vs sign(fwd), compared with what pure chance would give given how often each is up."""
    sig, fwd = np.asarray(sig), np.asarray(fwd)
    keep = fwd != 0
    sig, fwd = sig[keep], fwd[keep]
    n = len(sig)
    if n == 0:
        return {"n": 0, "hit": None, "expected": None, "z": 0.0}
    hit = float((sig == fwd).mean())
    p_up, p_bull = float((fwd > 0).mean()), float((sig > 0).mean())
    exp = p_up * p_bull + (1 - p_up) * (1 - p_bull)
    z = (hit - exp) / math.sqrt(exp * (1 - exp) / n) if 0 < exp < 1 else 0.0
    return {"n": n, "hit": hit, "expected": exp, "z": z}


def analyse(rows):
    dates = sorted({r["date"] for r in rows})
    cut = dates[int(len(dates) * 0.7)] if len(dates) >= 10 else None
    names = sorted({k for r in rows for k in r["parts"]})
    out, mult = [], {}
    for name in names:
        rec = {"name": name}
        sel = [r for r in rows if abs(r["parts"].get(name, 0)) >= 0.1]
        for h in HORIZONS:
            st = stats([np.sign(r["parts"][name]) for r in sel], [np.sign(r["fwd"][h]) for r in sel])
            rec[f"n{h}"], rec[f"hit{h}"], rec[f"z{h}"] = st["n"], st["hit"], st["z"]
            if h == 30:
                rec["expected30"] = st["expected"]
        # out-of-sample consistency (30 min)
        if cut:
            tr = [r for r in sel if r["date"] < cut]
            te = [r for r in sel if r["date"] >= cut]
            s_tr = stats([np.sign(r["parts"][name]) for r in tr], [np.sign(r["fwd"][30]) for r in tr])
            s_te = stats([np.sign(r["parts"][name]) for r in te], [np.sign(r["fwd"][30]) for r in te])
            e_tr = (s_tr["hit"] - s_tr["expected"]) if s_tr["n"] else 0
            e_te = (s_te["hit"] - s_te["expected"]) if s_te["n"] else 0
            rec["train_edge"], rec["test_edge"] = e_tr, e_te
            consistent = e_tr * e_te > 0
        else:
            consistent = False
        edge = (rec["hit30"] - rec["expected30"]) if rec.get("hit30") is not None else 0
        rec["multiplier"] = 1.0
        if rec["n30"] >= MIN_N and abs(rec["z30"]) >= MIN_Z and consistent:
            rec["multiplier"] = round(1 + max(-MAX_TWEAK, min(MAX_TWEAK, edge * 6)), 2)
            mult[name] = rec["multiplier"]
        out.append(rec)

    total = []
    for thr in (0.10, 0.18, 0.25, 0.35):
        sel = [r for r in rows if abs(r["score"]) >= thr]
        rec = {"min_score": thr}
        for h in HORIZONS:
            st = stats([np.sign(r["score"]) for r in sel], [np.sign(r["fwd"][h]) for r in sel])
            rec[f"n{h}"], rec[f"hit{h}"], rec[f"z{h}"] = st["n"], st["hit"], st["z"]
        total.append(rec)
    return out, total, mult, dates


def run(m5, daily):
    rows = collect(m5, daily)
    if len(rows) < 300:
        return None, {}
    ind, total, mult, dates = analyse(rows)
    report = {"generated_at": ist_now().isoformat(), "days": len(dates), "from": str(dates[0]), "to": str(dates[-1]),
              "samples": len(rows), "indicators": ind, "total_score": total, "adjusted": mult,
              "note": "Hit rates compare each indicator's lean with the next 15/30/60-minute move. "
                      "z above about 2.5 means the edge is unlikely to be luck. Weights change only for strong, "
                      "consistent edges, by at most 50%."}
    return report, mult


if __name__ == "__main__":
    m5, daily = fetch()
    if m5 is None:
        log("Could not download Nifty 5-minute data; nothing changed.")
        raise SystemExit(0)
    report, mult = run(m5, daily)
    if report is None:
        log("Not enough data for a backtest; nothing changed.")
        raise SystemExit(0)
    os.makedirs(DATA_DIR, exist_ok=True)
    save_json(os.path.join(DATA_DIR, "backtest.json"), report)
    save_json(os.path.join(DATA_DIR, "weights.json"), {"generated_at": report["generated_at"], "multipliers": mult})
    log(f"Backtest done: {report['samples']} samples over {report['days']} days; adjusted indicators: {mult or 'none'}")
    for r in report["indicators"]:
        print(f"  {r['name']:<28} n={r['n30']:<5} hit30={r['hit30'] if r['hit30'] is None else round(r['hit30']*100,1)}%  z={r['z30']:+.1f}  x{r['multiplier']}")
