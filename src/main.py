"""Entry point.  python -m src.main            (normal scheduled run)
                python -m src.main --force    (run outside market hours for testing; records nothing)"""
import argparse
import datetime as dt
import json
import os

from . import alerts, config, engine, events, gemini, indicators, market_data, news, options, scoreboard
from .config import EOD_CUTOFF, HORIZONS, IST, MARKET_CLOSE, MARKET_OPEN
from .nse import NSE
from .util import ist_now, load_json, log, save_json


class LiveSources:
    def __init__(self):
        self.nse = NSE()

    def nifty(self, now):
        return market_data.fetch_nifty()

    def global_(self, today):
        return market_data.fetch_global(today)

    def breadth(self, today, nifty_pct):
        return market_data.fetch_breadth(today, nifty_pct)

    def chain(self):
        c = self.nse.fetch_chain("NIFTY")
        return c, list(self.nse.notes[-3:])

    def news(self):
        return news.collect()


def market_phase(now):
    if now.weekday() >= 5:
        return "closed"
    t = now.time()
    if MARKET_OPEN <= t < MARKET_CLOSE:
        return "live"
    if MARKET_CLOSE <= t <= EOD_CUTOFF:
        return "eod"
    return "closed"


def P(name):
    return os.path.join(config.DATA_DIR, name)


def run(now, src, force=False):
    phase = market_phase(now)
    log(f"phase={phase} time={now:%Y-%m-%d %H:%M} IST")
    if phase == "closed" and not force:
        return "closed"
    if phase == "eod" and force and not os.path.exists(P("latest.json")):
        phase = "test"          # first ever run right after the close: build a full forecast page

    m1, daily = src.nifty(now)
    if m1 is None or m1.empty:
        log("No Nifty data from Yahoo; skipping this run.")
        return "no-data"

    latest_prev = load_json(P("latest.json"), {})
    history = load_json(P("history.json"), [])
    state = load_json(P("state.json"), {})
    daily_scores = load_json(P("daily_scores.json"), {})

    data_date = m1.index[-1].date()
    today = now.date()
    if phase in ("live", "eod") and data_date != today and not force:
        log("No candles for today (holiday or feed not started).")
        if now.time() > dt.time(9, 40):
            latest_prev.update(market_status="holiday", updated_at=now.isoformat())
            save_json(P("latest.json"), latest_prev)
        return "no-bars"

    resolved = scoreboard.resolve(history, m1, now)
    log(f"resolved {resolved} prediction horizons")

    # ------------------------------------------------ after the close: finalize scoreboard
    if phase == "eod":
        board = scoreboard.build(history, daily_scores, today.isoformat())
        history = scoreboard.prune(history, today)
        waiting = scoreboard.pending_due(history, now)
        if state.get("eod_sent") != today.isoformat() and (waiting == 0 or now.time() >= dt.time(16, 0)):
            if alerts.send_text(scoreboard.eod_text(board)):
                log("EOD scoreboard sent to Telegram")
            state["eod_sent"] = today.isoformat()
        latest_prev.update(market_status="closed", updated_at=now.isoformat())
        save_json(P("latest.json"), latest_prev)
        save_json(P("scoreboard.json"), {"generated_at": now.isoformat(), **board})
        save_json(P("history.json"), history)
        save_json(P("daily_scores.json"), daily_scores)
        save_json(P("state.json"), state)
        return "eod"

    # ------------------------------------------------ live run
    m5 = indicators.resample_5m(m1)
    price = float(m1["Close"].iloc[-1])
    anchor = m1.index[-1].to_pydatetime() + dt.timedelta(minutes=1)
    delay = max(0.0, (now - anchor).total_seconds() / 60)

    tuned = load_json(P("weights.json"), {}).get("multipliers", {})
    ind, tparts, tscore = indicators.compute(m5, daily, data_date, mult=tuned)
    glob, gscore = src.global_(data_date)
    vix = next((g["price"] for g in glob if g["symbol"] == "^INDIAVIX"), None)

    chain, nse_notes = src.chain()
    an, oscore, oparts = None, None, []
    if chain:
        an = options.analyze(chain, chain["spot"], state.get("oi_snapshot"), now)
        oscore, oparts = options.score(an)

    prev_close = (ind.get("pivots") or {}).get("prev_close")
    change = price - prev_close if prev_close else None
    change_pct = change / prev_close * 100 if prev_close else None

    b_items, bscore, bmeta = src.breadth(data_date, change_pct)
    items = src.news()
    ctx = f"Nifty {price:.0f}" + (f" ({change_pct:+.2f}% today)" if change_pct is not None else "") + \
          (f", India VIX {vix:.1f}" if vix else "") + f", technical bias {tscore:+.2f}" + \
          (f", options bias {oscore:+.2f}" if oscore is not None else "")
    nres = news.analyze(items, ctx, now)

    scores = {"technical": tscore, "options": oscore, "global": gscore, "news": nres["score"], "breadth": bscore}
    ev = events.resolve(data_date, an, nres)
    calib = scoreboard.calibration(history)
    preds = engine.build_prediction(anchor, price, scores, m5, vix, calib, event=ev["impact"])
    record = {"id": now.strftime("%Y-%m-%dT%H:%M"), "made_at": now.isoformat(), "anchor": anchor.isoformat(),
              "date": data_date.isoformat(), "spot": price, "scores": scores, "h": preds}

    parts = {"technical": tparts, "options": oparts, "breadth": (bmeta or {}).get("parts", [])}
    reasoning = engine.rule_reasoning(scores, parts, preds, ind, an, glob, nres, bmeta, ev)
    g_payload = json.dumps({
        "time_ist": now.strftime("%H:%M"), "nifty": round(price, 1), "change_pct_today": change_pct,
        "scores(-1..+1)": {k: (round(v, 2) if v is not None else None) for k, v in scores.items()},
        "predictions": {h: {k: (round(v, 1) if isinstance(v, float) else v) for k, v in p.items()
                            if k in ("dir", "conf_label", "low", "high", "mid")} for h, p in preds.items()},
        "technical_top": [f"{t['name']}: {t['note']}" for t in sorted(tparts, key=lambda x: -abs(x['value'] * x['weight']))[:5]],
        "options": ({"pcr": an["pcr"], "support": an["support"], "resistance": an["resistance"], "max_pain": an["max_pain"],
                     "atm_iv": an["atm_iv"], "dte": an["dte"]} if an else "unavailable"),
        "global": [f"{g['label']} {g['change_pct']:+.2f}%" for g in glob],
        "news": nres.get("summary"), "vix": vix,
        "heavyweights": ({"avg_pct": bmeta["heavy_pct"], "bank_nifty_pct": bmeta["bank_pct"], "note": bmeta["note"]} if bmeta else "unavailable"),
        "event_risk_today": ev})
    g_reason = engine.gemini_reasoning(g_payload)
    if g_reason:
        reasoning = g_reason

    # alerts (checked against predictions made BEFORE this run)
    record_run = phase == "live"
    pending = alerts.check_pending(history, price, now)
    price_15 = scoreboard.price_at(m1, anchor - dt.timedelta(minutes=15))
    sigma15 = {"move": (price - price_15) if price_15 else None, "sigma": engine.sigma_points(m5, vix, price, 15)}
    same_day = [r for r in history if r["date"] == data_date.isoformat()]
    prev_dir30 = same_day[-1]["h"]["30"]["dir"] if same_day else None
    new_alerts = [] if not record_run else alerts.detect(now, state, an=an, price=price, sigma15=sigma15, vix=vix, prev_vix=state.get("prev_vix"),
                               prev_dir30=prev_dir30, new_dir30=preds["30"]["dir"], new_conf30=preds["30"]["conf"],
                               news=nres, pending=pending)
    alerts.prune_keys(state, now)
    if state.get("alerts_date") != data_date.isoformat():
        state["alerts_today"], state["alerts_date"] = [], data_date.isoformat()
    state["alerts_today"] = (new_alerts + state["alerts_today"])[:40]
    sent = alerts.send_telegram([a for a in new_alerts if a["level"] in ("warn", "critical")]) if record_run else False
    log(f"{len(new_alerts)} new alerts (telegram sent: {sent})")

    if record_run:
        history.append(record)
        if chain:
            state["oi_snapshot"] = options.snapshot(chain, an, now)
        if vix:
            state["prev_vix"] = vix
    history = scoreboard.prune(history, today)
    board = scoreboard.build(history, daily_scores, data_date.isoformat())

    statuses = [p["status"] for p in pending]
    health = ("failing" if "failing" in statuses else "drifting" if "drifting" in statuses
              else "on_track" if statuses else "none")

    notes = []
    if delay > 20:
        notes.append(f"Yahoo data is about {delay:.0f} minutes behind.")
    if not chain:
        notes.append("NSE option chain unavailable this run; forecast uses technicals, global cues and news only.")
    if not glob:
        notes.append("Global market data unavailable this run.")
    if bscore is None:
        notes.append("Heavyweight-stock data unavailable this run.")
    if nres.get("source") in ("keywords", "none"):
        notes.append("News is scored with a simple keyword method (no Gemini result).")
    gap = abs(chain["spot"] / price - 1) * 100 if chain else None
    if gap and gap > 0.15:
        notes.append(f"Yahoo price and NSE spot differ by {gap:.2f}%: the price feed may be lagging.")

    day_hi = ind.get("day_high", price)
    latest = {
        "updated_at": now.isoformat(), "market_status": "live" if record_run else "test-run",
        "spot": {"price": price, "prev_close": prev_close, "change": change, "change_pct": change_pct,
                 "open": ind.get("day_open"), "high": day_hi, "low": ind.get("day_low"), "as_of": anchor.isoformat()},
        "prediction": {"anchor": anchor.isoformat(), "spot": price, "horizons": preds, "calibration": calib},
        "prediction_health": {"status": health, "pending": pending},
        "reasoning": reasoning,
        "scores": {k: {"value": v, "parts": parts.get(k, [])} for k, v in scores.items()},
        "weights": {str(h): config.WEIGHTS[h] for h in HORIZONS},
        "indicators": ind,
        "options": ({k: v for k, v in an.items() if k not in ("deltas",)} | {
            "movers": [{"strike": d["strike"], "side": d["side"], "d": d["d"], "pct": d["pct"], "z": d["z"]}
                       for d in sorted(an["deltas"], key=lambda x: -abs(x["d"]))[:6]]} if an else None),
        "global": glob,
        "breadth": {"items": b_items, "score": bscore, **(bmeta or {})},
        "event": ev,
        "tuned_weights": bool(tuned),
        "news": {**nres, "headlines": [{"title": i["title"], "source": i["source"],
                                        "time": i["time"].strftime("%H:%M") if i["time"] else ""} for i in items[:15]]},
        "alerts": state["alerts_today"],
        "series": [[int(ts.timestamp()), round(float(c), 2)] for ts, c in m5["Close"].tail(110).items()],
        "data_health": {"yahoo_delay_min": round(delay, 1), "nse_ok": bool(chain), "nse_notes": nse_notes,
                        "nse_spot": chain["spot"] if chain else None, "news_items": len(items),
                        "news_source": nres.get("source"), "reasoning_source": reasoning.get("source"), "notes": notes},
    }
    save_json(P("latest.json"), latest)
    save_json(P("scoreboard.json"), {"generated_at": now.isoformat(), **board})
    save_json(P("history.json"), history)
    save_json(P("daily_scores.json"), daily_scores)
    save_json(P("state.json"), state)
    log(f"done: 30m call {preds['30']['dir']} ({preds['30']['conf_label']}), health={health}")
    return "ok"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="run even when the market is closed (no predictions recorded)")
    args = ap.parse_args()
    run(ist_now(), LiveSources(), force=args.force)
