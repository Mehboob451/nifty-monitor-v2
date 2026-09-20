"""Option-chain analysis: PCR, max pain, OI walls, change since last run, and an options score."""
import datetime as dt

import numpy as np
import pandas as pd

from .config import IST
from .util import clip, sign


def analyze(chain, spot, prev, now):
    df = pd.DataFrame(chain["rows"]).sort_values("strike").reset_index(drop=True)
    tot_ce, tot_pe = float(df.ce_oi.sum()), float(df.pe_oi.sum())
    chg_ce, chg_pe = float(df.ce_chg.sum()), float(df.pe_chg.sum())
    pcr = tot_pe / tot_ce if tot_ce > 0 else None

    # Max pain: expiry price that hurts option buyers the most
    strikes = df.strike.values
    pain = [float(((np.maximum(p - strikes, 0) * df.ce_oi.values) + (np.maximum(strikes - p, 0) * df.pe_oi.values)).sum())
            for p in strikes]
    max_pain = float(strikes[int(np.argmin(pain))]) if len(pain) else None

    above, below = df[df.strike >= spot], df[df.strike <= spot]
    res = above.loc[above.ce_oi.idxmax()] if not above.empty and above.ce_oi.max() > 0 else None
    sup = below.loc[below.pe_oi.idxmax()] if not below.empty and below.pe_oi.max() > 0 else None
    top_ce = df.nlargest(3, "ce_oi")[["strike", "ce_oi"]].values.tolist()
    top_pe = df.nlargest(3, "pe_oi")[["strike", "pe_oi"]].values.tolist()

    atm = df.iloc[(df.strike - spot).abs().argsort()[:1]].iloc[0]
    ivs = [v for v in (atm.ce_iv, atm.pe_iv) if v and v > 0]
    atm_iv = float(np.mean(ivs)) if ivs else None

    try:
        dte = (dt.datetime.strptime(chain["expiry"], "%d-%b-%Y").date() - now.date()).days
    except (TypeError, ValueError):
        dte = None

    out = {"expiry": chain["expiry"], "dte": dte, "spot": spot, "pcr": pcr,
           "total_ce_oi": tot_ce, "total_pe_oi": tot_pe, "day_chg_ce": chg_ce, "day_chg_pe": chg_pe,
           "max_pain": max_pain, "atm_strike": float(atm.strike), "atm_iv": atm_iv,
           "resistance": {"strike": float(res.strike), "oi": float(res.ce_oi)} if res is not None else None,
           "support": {"strike": float(sup.strike), "oi": float(sup.pe_oi)} if sup is not None else None,
           "top_call_oi": top_ce, "top_put_oi": top_pe}

    # Change since the previous run (only if that snapshot is fresh and same expiry)
    fresh = False
    deltas = []
    if prev and prev.get("expiry") == chain["expiry"] and prev.get("date") == now.date().isoformat():
        age = (now - dt.datetime.fromisoformat(prev["time"])).total_seconds() / 60
        fresh = 0 < age <= 50
    out["delta_ok"] = fresh
    d_ce = d_pe = 0.0
    if fresh:
        p = prev["oi"]
        win = df[(df.strike >= spot - 1000) & (df.strike <= spot + 1000)]
        ce_d, pe_d = [], []
        for _, r in win.iterrows():
            po = p.get(str(int(r.strike)))
            if po is None:
                continue
            ce_d.append((r.strike, po[0], r.ce_oi, r.ce_oi - po[0]))
            pe_d.append((r.strike, po[1], r.pe_oi, r.pe_oi - po[1]))
        for side, arr, tot in (("CALL", ce_d, tot_ce), ("PUT", pe_d, tot_pe)):
            if not arr:
                continue
            vals = np.array([a[3] for a in arr], dtype=float)
            med = float(np.median(vals))
            mad = float(np.median(np.abs(vals - med))) * 1.4826 or float(vals.std()) or 1.0
            for k, po, no, d in arr:
                deltas.append({"strike": k, "side": side, "prev": po, "now": no, "d": d,
                               "pct": (d / po) if po > 0 else 0.0, "z": (d - med) / mad,
                               "share": abs(d) / tot if tot > 0 else 0.0})
        d_ce = float(sum(a[3] for a in ce_d))
        d_pe = float(sum(a[3] for a in pe_d))
        out["prev_pcr"] = prev.get("pcr")
        out["prev_res"] = prev.get("res")
        out["prev_sup"] = prev.get("sup")
    out["delta_ce"], out["delta_pe"], out["deltas"] = d_ce, d_pe, deltas

    # Compact per-strike data for the dashboard chart
    dmap = {(x["strike"], x["side"]): x["d"] for x in deltas}
    chart = df[(df.strike >= spot - 700) & (df.strike <= spot + 700)]
    out["chart"] = [[float(r.strike), float(r.ce_oi), float(r.pe_oi),
                     float(dmap.get((r.strike, "CALL"), 0)), float(dmap.get((r.strike, "PUT"), 0))]
                    for _, r in chart.iterrows()]
    return out


def snapshot(chain, an, now):
    return {"time": now.isoformat(), "date": now.date().isoformat(), "expiry": chain["expiry"], "pcr": an["pcr"],
            "res": (an["resistance"] or {}).get("strike"), "sup": (an["support"] or {}).get("strike"),
            "oi": {str(int(r["strike"])): [r["ce_oi"], r["pe_oi"]] for r in chain["rows"]}}


def score(an):
    """Options score in [-1, 1]: positive = bullish. Returns (score, parts)."""
    spot = an["spot"]
    parts = []

    def add(name, v, w, note):
        parts.append({"name": name, "value": round(float(clip(v)), 3), "weight": w, "note": note})

    if an["pcr"]:
        add("Put/Call ratio (OI)", (an["pcr"] - 1.0) / 0.35, 0.30,
            f"PCR {an['pcr']:.2f} " + ("(put writers dominate)" if an["pcr"] > 1.1 else "(call writers dominate)" if an["pcr"] < 0.9 else "(balanced)"))
    day_tot = abs(an["day_chg_pe"]) + abs(an["day_chg_ce"])
    if day_tot > 0:
        add("Today's OI build-up", (an["day_chg_pe"] - an["day_chg_ce"]) / day_tot, 0.20,
            "more put OI added than call OI" if an["day_chg_pe"] > an["day_chg_ce"] else "more call OI added than put OI")
    if an["delta_ok"]:
        dtot = abs(an["delta_pe"]) + abs(an["delta_ce"])
        if dtot > 0:
            add("Last 15-min OI flow", (an["delta_pe"] - an["delta_ce"]) / dtot, 0.15,
                "fresh put writing" if an["delta_pe"] > an["delta_ce"] else "fresh call writing")
    if an["resistance"] and an["support"]:
        dr = max(an["resistance"]["strike"] - spot, 1.0)
        ds = max(spot - an["support"]["strike"], 1.0)
        add("Distance to OI walls", (dr - ds) / (dr + ds), 0.20,
            f"support {an['support']['strike']:.0f} vs resistance {an['resistance']['strike']:.0f}")
    if an["max_pain"]:
        dte = an["dte"] if an["dte"] is not None else 5
        wt = 0.15 * (1.0 if dte <= 1 else 0.5 if dte <= 3 else 0.25)
        add("Max-pain pull", (an["max_pain"] - spot) / (0.004 * spot), wt, f"max pain {an['max_pain']:.0f}")
    if not parts:
        return None, []
    tw = sum(p["weight"] for p in parts)
    return clip(sum(p["value"] * p["weight"] for p in parts) / tw), parts
