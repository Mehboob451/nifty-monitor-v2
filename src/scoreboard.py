"""Resolves past predictions against what really happened and keeps the running scoreboard."""
import datetime as dt

import numpy as np
import pandas as pd

from .config import HISTORY_DAYS, HORIZONS
from .util import sign


# ---------------------------------------------------------------- resolving
def price_at(m1, t):
    """Nifty price at time t (close of the 1-minute bar ending at t). None if data has not caught up yet."""
    edge = pd.Timestamp(t) - pd.Timedelta(minutes=1)
    if m1.empty or m1.index[-1] < edge:
        return None
    sub = m1.loc[:edge]
    if sub.empty:
        return None
    gap = edge - sub.index[-1]
    if gap > pd.Timedelta(minutes=5) and m1.index[-1] < pd.Timestamp(t) + pd.Timedelta(minutes=10):
        return None
    return float(sub["Close"].iloc[-1])


def resolve(history, m1, now):
    """Fill in the actual outcome for every prediction whose target time has passed."""
    done = 0
    for rec in history:
        for h in HORIZONS:
            e = rec["h"].get(str(h))
            if not e or not e.get("scored") or e.get("actual") is not None:
                continue
            target = dt.datetime.fromisoformat(e["target_at"])
            if target > now:
                continue
            px = price_at(m1, target)
            if px is None:
                continue
            move = px - rec["spot"]
            e["actual"], e["move"], e["err"] = px, move, px - e["mid"]
            e["in68"] = bool(e["low"] <= px <= e["high"])
            e["in90"] = bool(e["low90"] <= px <= e["high90"])
            if e["dir"] == "UP":
                e["dir_hit"] = bool(move > 0)
            elif e["dir"] == "DOWN":
                e["dir_hit"] = bool(move < 0)
            else:
                e["dir_hit"] = bool(abs(move) <= 1.0 * e["sigma"])
            e["resolved_at"] = now.isoformat()
            done += 1
    return done


def pending_due(history, now):
    """Predictions scored-eligible whose target time has passed but are still unresolved."""
    n = 0
    for rec in history:
        for e in rec["h"].values():
            if e.get("scored") and e.get("actual") is None and dt.datetime.fromisoformat(e["target_at"]) <= now:
                n += 1
    return n


# ---------------------------------------------------------------- counters
def blank():
    return {"n": 0, "dir_hit": 0, "r68": 0, "r90": 0, "abs_err": 0.0,
            "conf": {"Low": [0, 0], "Medium": [0, 0], "High": [0, 0]},
            "by_dir": {"UP": [0, 0], "DOWN": [0, 0], "SIDEWAYS": [0, 0]}}


def add(c, e):
    if e.get("dir_hit") is None:
        return
    c["n"] += 1
    c["dir_hit"] += int(e["dir_hit"])
    c["r68"] += int(bool(e["in68"]))
    c["r90"] += int(bool(e["in90"]))
    c["abs_err"] += abs(e["err"])
    for bucket, key in ((c["conf"], e["conf_label"]), (c["by_dir"], e["dir"])):
        bucket[key][0] += 1
        bucket[key][1] += int(e["dir_hit"])


def merge(a, b):
    for k in ("n", "dir_hit", "r68", "r90", "abs_err"):
        a[k] += b[k]
    for grp in ("conf", "by_dir"):
        for k, (n, hit) in b[grp].items():
            a[grp][k][0] += n
            a[grp][k][1] += hit
    return a


def pct(x, n):
    return round(100 * x / n, 1) if n else None


def finalize(c):
    up, dn, sd = c["by_dir"]["UP"], c["by_dir"]["DOWN"], c["by_dir"]["SIDEWAYS"]
    return {"n": c["n"], "right": c["dir_hit"], "right_pct": pct(c["dir_hit"], c["n"]),
            "dir_calls": up[0] + dn[0], "dir_right": up[1] + dn[1], "dir_right_pct": pct(up[1] + dn[1], up[0] + dn[0]),
            "side_calls": sd[0], "side_held": sd[1], "side_held_pct": pct(sd[1], sd[0]),
            "in_range": c["r68"], "in_range_pct": pct(c["r68"], c["n"]), "in_range90_pct": pct(c["r90"], c["n"]),
            "mae": round(c["abs_err"] / c["n"], 1) if c["n"] else None,
            "conf": {k: {"n": v[0], "right_pct": pct(v[1], v[0])} for k, v in c["conf"].items()},
            "by_dir": {k: {"n": v[0], "right_pct": pct(v[1], v[0])} for k, v in c["by_dir"].items()}}


def day_counters(history, date):
    res = {str(h): blank() for h in HORIZONS}
    res["all"] = blank()
    for rec in history:
        if rec["date"] != date:
            continue
        for h in HORIZONS:
            e = rec["h"].get(str(h))
            if e:
                add(res[str(h)], e)
                add(res["all"], e)
    return res


# ---------------------------------------------------------------- learning from history
def calibration(history):
    """Sigma scale per horizon so that ~68% of outcomes land inside the predicted range."""
    out = {}
    for h in HORIZONS:
        z = [abs(r["h"][str(h)]["err"]) / r["h"][str(h)]["sigma_raw"]
             for r in history if str(h) in r["h"] and r["h"][str(h)].get("err") is not None
             and r["h"][str(h)].get("sigma_raw", 0) > 0][-150:]
        if len(z) >= 25:
            p68 = float(np.percentile(z, 68))
            out[str(h)] = round(float(min(1.8, max(0.5, 1 + (p68 - 1) * len(z) / (len(z) + 40)))), 3)
        else:
            out[str(h)] = 1.0
    return out


def component_accuracy(history, h="30"):
    comp = {}
    for r in history:
        e = r["h"].get(h)
        if not e or e.get("move") is None or abs(e["move"]) < 1e-9:
            continue
        for k, v in (r.get("scores") or {}).items():
            if v is None or abs(v) < 0.1:
                continue
            c = comp.setdefault(k, [0, 0])
            c[0] += 1
            c[1] += int(sign(v) == sign(e["move"]))
    return {k: {"n": n, "right_pct": pct(hit, n)} for k, (n, hit) in comp.items()}


# ---------------------------------------------------------------- output
def compact(rec):
    out = {"t": dt.datetime.fromisoformat(rec["made_at"]).strftime("%H:%M"), "spot": rec["spot"], "h": {}}
    for h in HORIZONS:
        e = rec["h"].get(str(h))
        if not e:
            continue
        out["h"][str(h)] = {"dir": e["dir"], "conf": round(e["conf"]), "conf_label": e["conf_label"],
                            "low": e["low"], "high": e["high"], "mid": e["mid"], "actual": e["actual"],
                            "dir_hit": e["dir_hit"], "in68": e["in68"], "scored": e["scored"],
                            "target": dt.datetime.fromisoformat(e["target_at"]).strftime("%H:%M")}
    return out


def build(history, daily_scores, today):
    for d in {r["date"] for r in history}:
        daily_scores[d] = day_counters(history, d)

    overall = {str(h): blank() for h in HORIZONS}
    overall["all"] = blank()
    for d, cnts in daily_scores.items():
        for k in overall:
            merge(overall[k], cnts[k])

    days = []
    for d in sorted(daily_scores)[-25:]:
        c = daily_scores[d]
        days.append({"date": d, "all": finalize(c["all"]), **{h: finalize(c[h]) for h in map(str, HORIZONS)}})

    t = daily_scores.get(today) or day_counters(history, today)
    return {
        "today": {"date": today, "all": finalize(t["all"]), **{h: finalize(t[h]) for h in map(str, HORIZONS)},
                  "records": [compact(r) for r in history if r["date"] == today]},
        "overall": {k: finalize(v) for k, v in overall.items()},
        "days": days[::-1],
        "components": component_accuracy(history),
        "calibration": calibration(history),
        "days_tracked": len(daily_scores),
    }


def prune(history, today):
    cutoff = (today - dt.timedelta(days=HISTORY_DAYS)).isoformat()
    return [r for r in history if r["date"] >= cutoff]


def eod_text(board):
    t = board["today"]
    a = t["all"]
    if not a["n"]:
        return "Nifty monitor: no scored predictions today."
    lines = [f"📊 <b>Nifty Monitor scoreboard {t['date']}</b>",
             f"Up/Down calls right: <b>{a['dir_right']}/{a['dir_calls']}</b> ({a['dir_right_pct']}%)  (coin flip = 50%)",
             f"Sideways calls that held: <b>{a['side_held']}/{a['side_calls']}</b>",
             f"Price landed inside predicted range: <b>{a['in_range']}/{a['n']}</b> ({a['in_range_pct']}%)  (target ~68%)"]
    for h in map(str, HORIZONS):
        s = t[h]
        if s["n"]:
            lines.append(f"  +{h} min: {s['dir_right']}/{s['dir_calls']} up/down right, {s['in_range_pct']}% in range")
    o = board["overall"]["all"]
    if o["n"] and o["n"] != a["n"]:
        lines.append(f"All-time ({board['days_tracked']} days): {o['dir_right_pct']}% up/down right, {o['in_range_pct']}% in range")
    return "\n".join(lines)
