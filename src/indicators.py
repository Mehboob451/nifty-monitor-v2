"""Technical indicators (pure pandas/numpy, no TA library needed)."""
import datetime as dt

import numpy as np
import pandas as pd

from .util import clip


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(close, n=14):
    d = close.diff()
    au = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    ad = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    out = 100 - 100 / (1 + au / ad.replace(0, np.nan))
    out[(ad == 0) & (au > 0)] = 100.0
    return out.fillna(50.0)


def true_range(df):
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()], axis=1).max(axis=1)
    return tr.fillna(df["High"] - df["Low"])


def atr(df, n=14):
    return true_range(df).ewm(alpha=1 / n, adjust=False).mean()


def adx(df, n=14):
    up = df["High"].diff()
    dn = -df["Low"].diff()
    plus = np.where((up > dn) & (up > 0), up, 0.0)
    minus = np.where((dn > up) & (dn > 0), dn, 0.0)
    a = atr(df, n)
    pdi = 100 * pd.Series(plus, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    mdi = 100 * pd.Series(minus, index=df.index).ewm(alpha=1 / n, adjust=False).mean() / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean().fillna(0), pdi.fillna(0), mdi.fillna(0)


def supertrend(df, n=10, mult=3.0):
    a = atr(df, n).values
    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    hl2 = (h + l) / 2
    ub, lb = hl2 + mult * a, hl2 - mult * a
    fub, flb = ub.copy(), lb.copy()
    d = np.ones(len(df))
    st = np.full(len(df), np.nan)
    for i in range(1, len(df)):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if d[i - 1] == 1 and c[i] < flb[i]:
            d[i] = -1
        elif d[i - 1] == -1 and c[i] > fub[i]:
            d[i] = 1
        else:
            d[i] = d[i - 1]
        st[i] = flb[i] if d[i] == 1 else fub[i]
    return pd.Series(st, index=df.index), pd.Series(d, index=df.index)


def stochastic(df, k=14, d=3):
    lo = df["Low"].rolling(k).min()
    hi = df["High"].rolling(k).max()
    kk = 100 * (df["Close"] - lo) / (hi - lo).replace(0, np.nan)
    return kk.fillna(50), kk.rolling(d).mean().fillna(50)


def resample_5m(m1):
    o = m1.resample("5min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    return o.dropna(subset=["Close"])


def pivots(daily, today):
    d = daily[daily.index.date < today]
    if d.empty:
        return None
    r = d.iloc[-1]
    h, l, c = float(r["High"]), float(r["Low"]), float(r["Close"])
    p = (h + l + c) / 3
    return {"P": p, "R1": 2 * p - l, "S1": 2 * p - h, "R2": p + (h - l), "S2": p - (h - l),
            "prev_high": h, "prev_low": l, "prev_close": c}


def compute(m5, daily, today, mult=None):
    """Returns (indicators dict, technical score parts list, technical score)."""
    c = m5["Close"]
    price = float(c.iloc[-1])
    ind = {"price": price}

    for n in (9, 21, 50):
        ind[f"ema{n}"] = float(ema(c, n).iloc[-1])
    ind["rsi"] = float(rsi(c).iloc[-1])
    line = ema(c, 12) - ema(c, 26)
    sig = ema(line, 9)
    hist = line - sig
    ind.update(macd=float(line.iloc[-1]), macd_signal=float(sig.iloc[-1]), macd_hist=float(hist.iloc[-1]),
               macd_hist_prev=float(hist.iloc[-2]) if len(hist) > 1 else 0.0)
    ma = c.rolling(20).mean()
    sd = c.rolling(20).std()
    up, lo = ma + 2 * sd, ma - 2 * sd
    ind.update(bb_upper=float(up.iloc[-1]), bb_lower=float(lo.iloc[-1]), bb_mid=float(ma.iloc[-1]))
    width = float(up.iloc[-1] - lo.iloc[-1])
    ind["bb_pctb"] = float((price - lo.iloc[-1]) / width) if width > 0 else 0.5
    a = atr(m5, 14)
    ind["atr"] = float(a.iloc[-1])
    ind["atr_pct"] = ind["atr"] / price * 100
    ax, pdi, mdi = adx(m5)
    ind.update(adx=float(ax.iloc[-1]), plus_di=float(pdi.iloc[-1]), minus_di=float(mdi.iloc[-1]))
    st, sdir = supertrend(m5)
    ind["supertrend"] = float(st.iloc[-1])
    ind["supertrend_dir"] = int(sdir.iloc[-1])
    k, d = stochastic(m5)
    ind.update(stoch_k=float(k.iloc[-1]), stoch_d=float(d.iloc[-1]))

    # Session VWAP. Yahoo gives no volume for the index, so fall back to a
    # time-weighted average price of the session (labelled honestly).
    day = m5[m5.index.date == today]
    if not day.empty:
        tp = (day["High"] + day["Low"] + day["Close"]) / 3
        if day["Volume"].sum() > 0:
            vw = float((tp * day["Volume"]).cumsum().iloc[-1] / day["Volume"].cumsum().iloc[-1])
            ind["vwap_kind"] = "VWAP"
        else:
            vw = float(tp.mean())
            ind["vwap_kind"] = "TWAP"
        ind["vwap"] = vw
        ind.update(day_open=float(day["Open"].iloc[0]), day_high=float(day["High"].max()), day_low=float(day["Low"].min()))
        first = day[day.index.time < dt.time(9, 30)]
        if len(day) >= 4 and not first.empty:
            orh, orl = float(first["High"].max()), float(first["Low"].min())
            ind.update(or_high=orh, or_low=orl,
                       or_state="above" if price > orh else ("below" if price < orl else "inside"))
    piv = pivots(daily, today)
    if piv:
        ind["pivots"] = piv

    parts = []
    mult = mult or {}

    def add(name, val, w, note=""):
        parts.append({"name": name, "value": round(float(clip(val)), 3), "weight": w * mult.get(name, 1.0), "note": note})

    adx_mult = 0.7 + 0.3 * min(ind["adx"] / 25.0, 1.0)

    t = (0.4 if price > ind["ema21"] else -0.4) + (0.3 if ind["ema9"] > ind["ema21"] else -0.3) \
        + (0.3 if ind["ema21"] > ind["ema50"] else -0.3)
    add("EMA trend (9/21/50)", t * adx_mult, 0.22,
        "price above 21-EMA with fast EMAs stacked up" if t > 0.5 else
        ("price below 21-EMA with EMAs stacked down" if t < -0.5 else "EMAs mixed"))
    if "vwap" in ind:
        add("Price vs " + ind["vwap_kind"], (price - ind["vwap"]) / (0.0015 * price) * adx_mult, 0.14,
            "above the session average" if price > ind["vwap"] else "below the session average")
    r = ind["rsi"]
    rs = clip((r - 50) / 18)
    if r >= 78:
        rs = -0.4
    elif r <= 22:
        rs = 0.4
    add("RSI(14)", rs, 0.12, f"RSI {r:.0f}" + (" (overbought, fade risk)" if r >= 70 else " (oversold, bounce risk)" if r <= 30 else ""))
    hs = clip(ind["macd_hist"] / (0.0004 * price)) * 0.7 + (0.3 if ind["macd_hist"] > ind["macd_hist_prev"] else -0.3)
    add("MACD", hs * adx_mult, 0.14, "histogram rising" if ind["macd_hist"] > ind["macd_hist_prev"] else "histogram falling")
    add("Supertrend", ind["supertrend_dir"] * adx_mult, 0.14, "bullish" if ind["supertrend_dir"] == 1 else "bearish")
    pb = ind["bb_pctb"]
    bs = clip((pb - 0.5) * 2) * 0.6
    if pb > 1.05:
        bs = -0.2
    elif pb < -0.05:
        bs = 0.2
    add("Bollinger position", bs, 0.06, f"%B {pb:.2f}")
    if piv:
        ps = 0.5 if price > piv["P"] else -0.5
        if abs(price - piv["R1"]) / price < 0.001:
            ps -= 0.3
        if abs(price - piv["S1"]) / price < 0.001:
            ps += 0.3
        add("Pivot levels", ps, 0.06, "above pivot" if price > piv["P"] else "below pivot")
    if "or_state" in ind:
        add("Opening range", {"above": 1.0, "below": -1.0, "inside": 0.0}[ind["or_state"]], 0.08,
            f"price {ind['or_state']} the first-15-minute range")
    ks = ind["stoch_k"]
    add("Stochastic", 0.3 if ks < 20 else (-0.3 if ks > 80 else (ks - 50) / 100), 0.04, f"%K {ks:.0f}")

    tw = sum(p["weight"] for p in parts)
    score = clip(sum(p["value"] * p["weight"] for p in parts) / tw)
    return ind, parts, score
