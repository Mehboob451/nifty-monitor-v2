"""World-news scan: Google News RSS + GDELT headlines, scored for Nifty impact by Gemini
(optionally with Google Search grounding). Falls back to a simple keyword score."""
import datetime as dt
import re
import time
from urllib.parse import quote

import feedparser
import requests

from . import gemini
from .config import IST, USE_GROUNDED_SEARCH
from .util import clip, log

QUERIES = [
    'Nifty OR Sensex OR "Indian stock market"',
    'RBI OR "Reserve Bank of India" OR "India inflation"',
    '"Federal Reserve" OR "Fed rate" OR "US inflation"',
    'crude oil OR Brent OR OPEC',
    'FII OR FPI India selling OR buying',
    'tariff OR sanctions OR war OR ceasefire OR "Middle East"',
    '"Wall Street" OR "Asian markets" OR "European stocks"',
    'India economy OR GDP OR "India budget" OR SEBI',
]
UA = {"User-Agent": "Mozilla/5.0 (compatible; nifty-monitor/1.0)"}


def _norm(t):
    return re.sub(r"[^a-z0-9 ]", "", t.lower())[:80]


def fetch_rss(query, hours=3, limit=8):
    url = f"https://news.google.com/rss/search?q={quote(query + f' when:{hours}h')}&hl=en-IN&gl=IN&ceid=IN:en"
    try:
        r = requests.get(url, headers=UA, timeout=20)
        feed = feedparser.parse(r.content)
    except Exception as e:  # noqa: BLE001
        log(f"rss failed ({query[:20]}): {type(e).__name__}")
        return []
    out = []
    for e in feed.entries[:limit]:
        ts = None
        if getattr(e, "published_parsed", None):
            ts = dt.datetime(*e.published_parsed[:6], tzinfo=dt.timezone.utc).astimezone(IST)
        src = e.get("source", {}).get("title") if isinstance(e.get("source"), dict) else None
        title = re.sub(r"\s+-\s+[^-]+$", "", e.title).strip()
        out.append({"title": title, "source": src or "", "time": ts, "link": e.get("link", "")})
    return out


def fetch_gdelt(minutes=120):
    q = quote('(india OR nifty OR "crude oil" OR "federal reserve" OR tariff) sourcelang:english')
    url = (f"https://api.gdeltproject.org/api/v2/doc/doc?query={q}&mode=artlist&maxrecords=25"
           f"&format=json&timespan={minutes}min&sort=datedesc")
    try:
        r = requests.get(url, headers=UA, timeout=25)
        arts = r.json().get("articles", [])
    except Exception:  # noqa: BLE001
        return []
    out = []
    for a in arts:
        try:
            ts = dt.datetime.strptime(a["seendate"], "%Y%m%dT%H%M%SZ").replace(tzinfo=dt.timezone.utc).astimezone(IST)
        except (KeyError, ValueError):
            ts = None
        out.append({"title": a.get("title", "").strip(), "source": a.get("domain", ""), "time": ts, "link": a.get("url", "")})
    return out


def collect(max_items=45):
    seen, items = set(), []
    batches = []
    for q in QUERIES:
        batches.append(fetch_rss(q))
        time.sleep(0.4)
    batches.append(fetch_gdelt())
    for batch in batches:
        for it in batch:
            k = _norm(it["title"])
            if it["title"] and k not in seen:
                seen.add(k)
                items.append(it)
    items.sort(key=lambda x: x["time"] or dt.datetime(2000, 1, 1, tzinfo=IST), reverse=True)
    return items[:max_items]


BULL = ["rally", "surge", "record high", "rate cut", "ceasefire", "inflow", "fii buying", "upgrade", "beats",
        "soars", "jumps", "stimulus", "trade deal", "gains", "rebound", "eases"]
BEAR = ["crash", "plunge", "sell-off", "selloff", "war", "attack", "sanction", "tariff", "rate hike", "outflow",
        "fii selling", "downgrade", "slump", "tumble", "recession", "default", "escalat", "drops", "falls", "slides"]


def keyword_score(items):
    b = r = 0
    for it in items:
        t = it["title"].lower()
        b += sum(k in t for k in BULL)
        r += sum(k in t for k in BEAR)
    score = clip((b - r) / (b + r + 6))
    return {"score": score, "impact": "medium" if (b + r) >= 10 else "low",
            "summary": f"Keyword scan of {len(items)} headlines ({b} positive, {r} negative terms). Add a Gemini key for real analysis.",
            "items": [], "source": "keywords"}


def _prompt(items, ctx, now):
    lines = "\n".join(f"{i + 1}. [{(it['time'].strftime('%H:%M') if it['time'] else '--')}] {it['title']} ({it['source']})"
                      for i, it in enumerate(items))
    return f"""You are a market analyst watching the Indian stock market (Nifty 50).
Current time: {now:%A %d %b %Y, %H:%M} IST. Market context: {ctx}

Below are the latest headlines (newest first). {"Also search the web yourself for any market-moving development in the last 2 hours that these headlines miss (global markets, central banks, geopolitics, crude oil, FII/DII flows, big Indian company or policy news)." if USE_GROUNDED_SEARCH else ""}

Judge only the likely effect on Nifty over the NEXT 60 MINUTES. Ignore stale, repeated or already-priced-in news. Most of the time the answer is close to 0; use |score| above 0.4 only for a clear, fresh, market-moving event.

Headlines:
{lines}

Reply with ONLY a JSON object, no markdown:
{{"score": number from -1 (very bearish) to 1 (very bullish),
 "impact": "low" | "medium" | "high",
 "summary": "2 sentences on what matters most right now",
 "event_risk": "none" | "moderate" | "high"  (is a major SCHEDULED event due today or in the next 2 hours, e.g. RBI or Fed decision, US CPI or jobs data, India CPI or GDP, Budget, big election result?),
 "event_note": "name and time of that event, or empty string",
 "items": [{{"title": "short headline", "direction": "bullish" | "bearish" | "neutral", "impact": "low" | "medium" | "high", "why": "one short sentence"}}]}}
Include at most 8 items, most important first."""


def analyze(items, ctx, now):
    if not items and not gemini.available():
        return {"score": None, "impact": "low", "summary": "No headlines available.", "items": [], "source": "none"}
    obj = model = None
    grounded = False
    if gemini.available():
        prompt = _prompt(items, ctx, now)
        if USE_GROUNDED_SEARCH:
            obj, model = gemini.ask_json(prompt, grounded=True)
            grounded = obj is not None
        if obj is None:
            obj, model = gemini.ask_json(prompt, grounded=False)
    try:
        if obj is not None:
            return {"score": clip(float(obj.get("score", 0))), "impact": str(obj.get("impact", "low")).lower(),
                    "summary": str(obj.get("summary", ""))[:400],
                    "event_risk": str(obj.get("event_risk", "none")).lower(), "event_note": str(obj.get("event_note", ""))[:160],
                    "items": [{"title": str(i.get("title", ""))[:160], "direction": str(i.get("direction", "neutral")).lower(),
                               "impact": str(i.get("impact", "low")).lower(), "why": str(i.get("why", ""))[:200]}
                              for i in (obj.get("items") or [])[:8]],
                    "source": f"{model}{' + Google Search' if grounded else ''}"}
    except (TypeError, ValueError):
        pass
    return keyword_score(items) if items else {"score": None, "impact": "low", "summary": "No headlines available.", "items": [], "source": "none"}
