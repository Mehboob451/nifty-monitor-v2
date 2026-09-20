"""Thin Gemini REST client (free tier). Tries several models; every failure is soft."""
import json
import os
import re

import requests

from .config import GEMINI_MODELS
from .util import log

URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"


def available():
    return bool(os.environ.get("GEMINI_API_KEY"))


def _text(resp_json):
    parts = resp_json["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts).strip()


def parse_json(text):
    if not text:
        return None
    t = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        return json.loads(t[a:b + 1])
    except ValueError:
        return None


def ask_json(prompt, grounded=False, temperature=0.2, timeout=120):
    """Returns (dict, model_name) or (None, None). grounded=True lets Gemini use Google Search."""
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        return None, None
    for model in GEMINI_MODELS:
        body = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"temperature": temperature}}
        if grounded:
            body["tools"] = [{"google_search": {}}]       # cannot be combined with JSON mode
        else:
            body["generationConfig"]["responseMimeType"] = "application/json"
        try:
            r = requests.post(URL.format(model=model), json=body, timeout=timeout,
                              headers={"x-goog-api-key": key, "Content-Type": "application/json"})
        except requests.RequestException as e:
            log(f"gemini {model}: {type(e).__name__}")
            continue
        if r.status_code != 200:
            log(f"gemini {model}: HTTP {r.status_code} {r.text[:120]!r}")
            continue
        try:
            obj = parse_json(_text(r.json()))
        except (KeyError, IndexError, ValueError):
            obj = None
        if obj:
            return obj, model
    return None, None
