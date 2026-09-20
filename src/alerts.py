"""Alert detection ('big fish' proxies, model-going-wrong warnings) and Telegram delivery."""
import datetime as dt
import html
import os

import requests

from .config import (ALERT_COOLDOWN_MIN, OI_MIN_PCT, OI_MIN_SHARE, OI_MIN_Z, PCR_SHIFT, PRICE_SHOCK_SIGMA,
                     VIX_SPIKE_PCT)
from .util import log


def _allow(state, key, now, minutes=ALERT_COOLDOWN_MIN, once=False):
    keys = state.setdefault("alert_keys", {})
    last = keys.get(key)
    if last:
        if once or (now - dt.datetime.fromisoformat(last)).total_seconds() / 60 < minutes:
            return False
    keys[key] = now.isoformat()
    return True


def prune_keys(state, now):
    keys = state.get("alert_keys", {})
    state["alert_keys"] = {k: v for k, v in keys.items()
                           if (now - dt.datetime.fromisoformat(v)).total_seconds() < 86400}


def check_pending(history, price, now):
    """Status of every still-open prediction: on_track / drifting / failing."""
    out = []
    for rec in history:
        for h, e in rec["h"].items():
            if not e.get("scored") or e.get("actual") is not None:
                continue
            target = dt.datetime.fromisoformat(e["target_at"])
            if target <= now:
                continue
            hm = int(h)
            elapsed = 1 - (target - now).total_seconds() / 60 / hm
            sig, spot, move = e["sigma"], rec["spot"], price - rec["spot"]
            status = "on_track"
            if price < e["low90"] or price > e["high90"] or (
                    elapsed >= 0.3 and (price < e["low"] - 0.25 * sig or price > e["high"] + 0.25 * sig)):
                status = "failing"
            elif elapsed >= 0.3 and ((e["dir"] == "UP" and move < -0.6 * sig) or (e["dir"] == "DOWN" and move > 0.6 * sig)
                                     or (e["dir"] == "SIDEWAYS" and abs(move) > 1.0 * sig)):
                status = "drifting"
            out.append({"id": rec["id"], "made": dt.datetime.fromisoformat(rec["made_at"]).strftime("%H:%M"),
                        "h": hm, "dir": e["dir"], "low": e["low"], "high": e["high"], "target": target.strftime("%H:%M"),
                        "spot": spot, "status": status})
    return out


def detect(now, state, *, an, price, sigma15, vix, prev_vix, prev_dir30, new_dir30, new_conf30, news, pending):
    alerts = []

    def add(level, typ, title, detail):
        alerts.append({"time": now.isoformat(), "level": level, "type": typ, "title": title, "detail": detail})

    # 1) Big money in the option chain: unusual OI change at a strike vs the last run
    if an and an.get("delta_ok"):
        movers = sorted(an["deltas"], key=lambda x: -abs(x["z"]))
        n = 0
        for d in movers:
            if n >= 3:
                break
            near = abs(d["strike"] - price) <= 150
            if d["d"] > 0 and d["z"] >= OI_MIN_Z and d["pct"] >= OI_MIN_PCT and d["share"] >= OI_MIN_SHARE:
                key = f"oi:{int(d['strike'])}:{d['side']}:add"
                if _allow(state, key, now):
                    if d["side"] == "CALL":
                        meaning = "usually call writing: a ceiling is being built" if d["strike"] >= price else "heavy call positions below spot: bearish hedge or writing"
                    else:
                        meaning = "usually put writing: a floor is being built" if d["strike"] <= price else "heavy put positions above spot: bullish bets or hedge"
                    add("critical" if near else "warn", "oi_spike", f"Big {d['side']} OI jump at {d['strike']:.0f}",
                        f"+{d['d']:,.0f} contracts in ~15 min ({d['pct'] * 100:.0f}% up); {meaning}.")
                    n += 1
            elif d["d"] < 0 and d["z"] <= -OI_MIN_Z and d["pct"] <= -OI_MIN_PCT and d["share"] >= OI_MIN_SHARE:
                key = f"oi:{int(d['strike'])}:{d['side']}:unwind"
                if _allow(state, key, now):
                    meaning = "resistance is weakening (bullish)" if d["side"] == "CALL" else "support is weakening (bearish)"
                    add("critical" if near else "warn", "oi_unwind", f"{d['side']} OI unwinding at {d['strike']:.0f}",
                        f"{d['d']:,.0f} contracts in ~15 min ({d['pct'] * 100:.0f}%); {meaning}.")
                    n += 1
        pcr, ppcr = an.get("pcr"), an.get("prev_pcr")
        if pcr and ppcr and abs(pcr - ppcr) >= PCR_SHIFT and _allow(state, "pcr", now):
            add("warn", "pcr_shift", f"PCR moved {ppcr:.2f} to {pcr:.2f}",
                "Option positioning turned more bullish." if pcr > ppcr else "Option positioning turned more bearish.")
        for name, cur, prev_ in (("Resistance", (an["resistance"] or {}).get("strike"), an.get("prev_res")),
                                 ("Support", (an["support"] or {}).get("strike"), an.get("prev_sup"))):
            if cur and prev_ and cur != prev_ and _allow(state, f"wall:{name}", now, minutes=60):
                add("info", "wall_shift", f"{name} wall moved {prev_:.0f} to {cur:.0f}", "Largest open-interest strike changed.")

    # 2) Sudden price or volatility shock
    if sigma15 and sigma15["move"] is not None:
        z = abs(sigma15["move"]) / sigma15["sigma"]
        if z >= PRICE_SHOCK_SIGMA and _allow(state, "shock", now, minutes=30):
            add("critical" if z >= 2.5 else "warn", "price_shock",
                f"Sharp {'rise' if sigma15['move'] > 0 else 'fall'}: {sigma15['move']:+.0f} pts in 15 min",
                f"That is {z:.1f}x the normal 15-minute move.")
    if vix and prev_vix and (vix / prev_vix - 1) * 100 >= VIX_SPIKE_PCT and _allow(state, "vix", now):
        add("warn", "vix_spike", f"India VIX jumped {prev_vix:.2f} to {vix:.2f}", "Fear is rising; expect wider swings.")

    # 3) Model flipped its mind
    if prev_dir30 in ("UP", "DOWN") and new_dir30 in ("UP", "DOWN") and prev_dir30 != new_dir30 and new_conf30 >= 30 \
            and _allow(state, "flip", now, minutes=30):
        add("warn", "bias_flip", f"Model flipped from {prev_dir30} to {new_dir30}", "Earlier calls may no longer hold.")

    # 4) Open predictions going wrong
    bad = [p for p in pending if p["status"] in ("failing", "drifting")
           and _allow(state, f"pend:{p['id']}:{p['h']}:{p['status']}", now, once=True)]
    if bad:
        worst = "critical" if any(p["status"] == "failing" for p in bad) else "warn"
        lines = [f"+{p['h']}m call ({p['dir']}) from {p['made']}: range {p['low']:.0f}-{p['high']:.0f}, "
                 f"now {price:.0f} ({p['status']})" for p in bad[:3]]
        add(worst, "prediction_wrong", "Prediction may be going wrong", " | ".join(lines))

    # 5) High-impact news
    if news and news.get("impact") == "high" and news.get("score") is not None and abs(news["score"]) >= 0.4 \
            and _allow(state, "news:" + news.get("summary", "")[:50], now, minutes=60):
        add("warn", "news", f"High-impact news ({'bullish' if news['score'] > 0 else 'bearish'})", news.get("summary", ""))
    return alerts


def send_telegram(alerts):
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat) or not alerts:
        return False
    icon = {"critical": "🚨", "warn": "⚠️", "info": "ℹ️"}
    text = "\n\n".join(f"{icon.get(a['level'], '')} <b>{html.escape(a['title'])}</b>\n{html.escape(a['detail'])}" for a in alerts[:6])
    return _post(token, chat, text)


def send_text(text):
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    return _post(token, chat, text) if token and chat else False


def _post(token, chat, text):
    try:
        r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", timeout=20,
                          json={"chat_id": chat, "text": text[:4000], "parse_mode": "HTML", "disable_web_page_preview": True})
        if r.status_code != 200:
            log(f"telegram HTTP {r.status_code}: {r.text[:120]}")
        return r.status_code == 200
    except requests.RequestException as e:
        log(f"telegram failed: {type(e).__name__}")
        return False
