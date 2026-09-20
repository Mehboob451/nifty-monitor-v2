import datetime as dt
import json
import math
import os

import numpy as np

from .config import IST


def ist_now():
    return dt.datetime.now(IST)


def clip(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))


def sign(x, eps=1e-9):
    return 1 if x > eps else (-1 if x < -eps else 0)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if (math.isnan(f) or math.isinf(f)) else round(f, 4)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (dt.datetime, dt.date)):
        return o.isoformat()
    return o


def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, ValueError):
        return default


def save_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(jsonable(obj), f, separators=(",", ":"), ensure_ascii=False)
    os.replace(tmp, path)


def log(msg):
    print(f"[{ist_now():%H:%M:%S}] {msg}", flush=True)
