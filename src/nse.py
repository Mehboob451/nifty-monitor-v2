"""NSE option chain (public website API). No key needed, but NSE often blocks cloud IPs,
so every step is defensive and the rest of the system keeps working if this fails."""
import datetime as dt
import time

import requests

from .config import IST
from .util import log

BASE = "https://www.nseindia.com"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
      "Chrome/124.0.0.0 Safari/537.36")


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def nearest_expiry(expiries):
    today = dt.datetime.now(IST).date()
    best = None
    for e in expiries or []:
        try:
            d = dt.datetime.strptime(e, "%d-%b-%Y").date()
        except ValueError:
            continue
        if d >= today and (best is None or d < best[0]):
            best = (d, e)
    return best[1] if best else None


def parse_chain(raw, expiry):
    rec = (raw or {}).get("records") or {}
    rows, spot = [], rec.get("underlyingValue")
    for r in rec.get("data") or []:
        ex = r.get("expiryDate") or r.get("expiryDates")
        if expiry and ex and ex != expiry:
            continue
        k = r.get("strikePrice")
        if k is None:
            continue
        ce, pe = r.get("CE") or {}, r.get("PE") or {}
        if spot is None:
            spot = ce.get("underlyingValue") or pe.get("underlyingValue")
        rows.append({
            "strike": float(k),
            "ce_oi": _num(ce.get("openInterest")), "ce_chg": _num(ce.get("changeinOpenInterest")),
            "ce_vol": _num(ce.get("totalTradedVolume")), "ce_iv": _num(ce.get("impliedVolatility")),
            "ce_ltp": _num(ce.get("lastPrice")),
            "pe_oi": _num(pe.get("openInterest")), "pe_chg": _num(pe.get("changeinOpenInterest")),
            "pe_vol": _num(pe.get("totalTradedVolume")), "pe_iv": _num(pe.get("impliedVolatility")),
            "pe_ltp": _num(pe.get("lastPrice")),
        })
    if not rows or not spot:
        return None
    return {"expiry": expiry, "spot": float(spot), "timestamp": rec.get("timestamp"), "rows": rows}


class NSE:
    def __init__(self):
        self.s = requests.Session()
        self.s.headers.update({
            "User-Agent": UA, "Accept": "application/json, text/plain, */*",
            "Accept-Language": "en-US,en;q=0.9", "Accept-Encoding": "gzip, deflate",
            "Referer": BASE + "/option-chain", "Connection": "keep-alive"})
        self.warmed = False
        self.notes = []

    def warm(self):
        for url in (BASE + "/", BASE + "/option-chain"):
            try:
                self.s.get(url, headers={"Accept": "text/html,application/xhtml+xml"}, timeout=20)
            except requests.RequestException as e:
                self.notes.append(f"warm-up {url}: {type(e).__name__}")
            time.sleep(0.8)
        self.warmed = True

    def get(self, path, params=None, tries=3):
        if not self.warmed:
            self.warm()
        for i in range(tries):
            try:
                r = self.s.get(BASE + path, params=params, timeout=20)
                if r.status_code == 200:
                    try:
                        data = r.json()
                    except ValueError:
                        data = None
                    if data:
                        return data
                    self.notes.append(f"{path}: empty response")
                else:
                    self.notes.append(f"{path}: HTTP {r.status_code}")
                    if r.status_code in (401, 403):
                        self.warm()
            except requests.RequestException as e:
                self.notes.append(f"{path}: {type(e).__name__}")
            time.sleep(1.5 * (i + 1))
        return None

    def fetch_chain(self, symbol="NIFTY"):
        # Newer endpoints first (need an explicit expiry), then the legacy one.
        info = self.get("/api/option-chain-contract-info", {"symbol": symbol}, tries=2)
        expiry = nearest_expiry((info or {}).get("expiryDates"))
        if expiry:
            raw = self.get("/api/option-chain-v3", {"type": "Indices", "symbol": symbol, "expiry": expiry}, tries=2)
            chain = parse_chain(raw, expiry)
            if chain:
                return chain
        raw = self.get("/api/option-chain-indices", {"symbol": symbol}, tries=2)
        if raw:
            chain = parse_chain(raw, nearest_expiry((raw.get("records") or {}).get("expiryDates")))
            if chain:
                return chain
        log("NSE option chain unavailable: " + "; ".join(self.notes[-4:]))
        return None
