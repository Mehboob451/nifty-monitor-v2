"""Nifty candles + global cues from Yahoo Finance (free, no key)."""
import time

import pandas as pd

from .config import BANK_NIFTY, GLOBAL_CUES, HEAVYWEIGHTS, IST
from .util import clip, log


def _retry(fn, tries=3, wait=3):
    last = None
    for i in range(tries):
        try:
            r = fn()
            if r is not None and len(r):
                return r
        except Exception as e:  # noqa: BLE001 - network flakiness is expected
            last = e
        time.sleep(wait * (i + 1))
    if last:
        log(f"yfinance failed: {last}")
    return None


def fetch_nifty():
    """Returns (1-minute candles for ~5 days, daily candles for ~1 year), both IST-indexed."""
    import yfinance as yf
    t = yf.Ticker("^NSEI")
    m1 = _retry(lambda: t.history(period="5d", interval="1m", auto_adjust=False))
    daily = _retry(lambda: t.history(period="1y", interval="1d", auto_adjust=False))
    if m1 is None or daily is None:
        return None, None
    m1.index = m1.index.tz_convert(IST) if m1.index.tz else m1.index.tz_localize(IST)
    daily.index = (daily.index.tz_convert(IST) if daily.index.tz else daily.index.tz_localize(IST)).normalize()
    return m1[["Open", "High", "Low", "Close", "Volume"]], daily[["Open", "High", "Low", "Close", "Volume"]]


def fetch_global(today):
    """Percent change of each global cue vs its previous close, plus a combined score in [-1, 1]."""
    import yfinance as yf
    out = []
    tickers = list(GLOBAL_CUES)
    df = _retry(lambda: yf.download(tickers, period="7d", interval="1d", group_by="ticker",
                                    progress=False, threads=False, auto_adjust=False))
    if df is None:
        return [], None
    num = den = 0.0
    for sym, (label, sgn, w, scale) in GLOBAL_CUES.items():
        try:
            close = df[sym]["Close"].dropna()
            if len(close) < 2:
                continue
            last, prev = float(close.iloc[-1]), float(close.iloc[-2])
            pct = (last / prev - 1) * 100
            fresh = close.index[-1].date() >= today   # is the latest bar from today?
            eff_w = w * (1.0 if fresh else 0.4)       # stale markets count less
            sig = clip(sgn * pct / scale)
            num += eff_w * sig
            den += eff_w
            out.append({"symbol": sym, "label": label, "price": last, "change_pct": pct,
                        "signal": sig, "stale": not fresh, "weight": eff_w})
        except Exception:  # noqa: BLE001
            continue
    return out, (clip(num / den) if den > 0 else None)


def fetch_breadth(today, nifty_pct):
    """Heavyweight stocks + Bank Nifty. Returns (items, score in [-1,1] or None, meta)."""
    import yfinance as yf
    syms = list(HEAVYWEIGHTS) + [BANK_NIFTY]
    df = _retry(lambda: yf.download(syms, period="7d", interval="1d", group_by="ticker",
                                    progress=False, threads=False, auto_adjust=False))
    if df is None:
        return [], None, None
    items, num, den, bank = [], 0.0, 0.0, None
    for sym in syms:
        try:
            close = df[sym]["Close"].dropna()
            if len(close) < 2 or close.index[-1].date() < today:
                continue
            pct = (float(close.iloc[-1]) / float(close.iloc[-2]) - 1) * 100
        except Exception:  # noqa: BLE001
            continue
        if sym == BANK_NIFTY:
            bank = pct
            items.append({"symbol": sym, "label": "Bank Nifty", "change_pct": pct, "weight": 0})
        else:
            label, w = HEAVYWEIGHTS[sym]
            num, den = num + w * pct, den + w
            items.append({"symbol": sym, "label": label, "change_pct": pct, "weight": w})
    if den < 0.5 * sum(w for _, w in HEAVYWEIGHTS.values()):
        return items, None, None
    hv = num / den
    lead = hv if bank is None else 0.5 * hv + 0.5 * bank
    parts = [{"name": "Heavyweights + Bank Nifty", "value": round(clip(lead / 0.6), 3), "weight": 0.65,
              "note": f"heavyweights {hv:+.2f}%" + (f", Bank Nifty {bank:+.2f}%" if bank is not None else "")}]
    score = clip(lead / 0.6)
    if nifty_pct is not None:
        gap = clip((lead - nifty_pct) / 0.35)
        parts.append({"name": "Leading or lagging the index", "value": round(gap, 3), "weight": 0.35,
                      "note": f"blue chips {lead:+.2f}% vs Nifty {nifty_pct:+.2f}%"})
        score = 0.65 * score + 0.35 * gap
    note = None
    if nifty_pct is not None:
        if nifty_pct > 0.15 and lead < -0.05:
            note = "Nifty is up but the heavyweights are down: the rise looks fragile."
        elif nifty_pct < -0.15 and lead > 0.05:
            note = "Nifty is down but the heavyweights are up: the fall may not last."
    return items, clip(score), {"heavy_pct": hv, "bank_pct": bank, "lead": lead, "parts": parts, "note": note}
