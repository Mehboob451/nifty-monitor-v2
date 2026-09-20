"""Event awareness: scheduled events (events.json you can edit), expiry day, and Gemini's live check."""
import os

from .util import load_json

LEVELS = ["none", "moderate", "high"]
PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "events.json")


def resolve(date, an, news):
    """Returns {"impact": none|moderate|high, "items": [text, ...]} for this trading day."""
    level, items = 0, []
    e = load_json(PATH, {}).get(date.isoformat())
    if isinstance(e, dict) and e.get("name"):
        items.append(str(e["name"]))
        level = max(level, LEVELS.index(e.get("impact")) if e.get("impact") in LEVELS else 1)
    if an and an.get("dte") == 0:
        items.append("Nifty weekly expiry day: option-driven whipsaws are more likely")
        level = max(level, 1)
    if news and news.get("event_risk") in ("moderate", "high") and news.get("event_note"):
        items.append(str(news["event_note"])[:140])
        level = max(level, 1)          # a live AI check can raise caution to "moderate", never to "high"
    return {"impact": LEVELS[level], "items": items}
