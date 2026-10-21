"""Structured data layer: products, pincodes, holidays, date maths and the provided mock tools."""
import difflib
import importlib.util
import re
from datetime import date, datetime, timedelta
from functools import lru_cache

import pandas as pd

from .config import DATA_DIR, TODAY, TOOLS_FILE


def _csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name, dtype=str, keep_default_na=False)


def _row(series) -> dict:
    """A CSV row as a dict with blanks dropped (blank = missing / not applicable)."""
    return {k: v for k, v in series.items() if v != ""}


# ---------------------------------------------------------------- provided tools
@lru_cache(maxsize=1)
def provided_tools():
    """Import lookup_order / create_ticket from the dataset's tools/tools.py by path."""
    spec = importlib.util.spec_from_file_location("zentromart_tools", TOOLS_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------- products
@lru_cache(maxsize=1)
def products() -> pd.DataFrame:
    return _csv("products.csv")


@lru_cache(maxsize=1)
def sku_prefixes() -> set:
    return {s.split("-")[0].upper() for s in products()["sku"] if "-" in s}


def _name_columns(df):
    return [c for c in df.columns if "name" in c.lower() or c.lower() in ("title", "brand", "model")]


def lookup_product(query: str) -> dict:
    """Find a product by exact SKU, else by closest name match (top 3)."""
    df = products()
    q = (query or "").strip()
    sku = re.sub(r"\s+", "", q).upper()
    m = re.match(r"^([A-Z]{2,5})-?(\d{2,6})$", sku)
    if m:
        sku = f"{m.group(1)}-{m.group(2)}"
    hit = df[df["sku"].str.upper() == sku]
    if len(hit):
        row = _row(hit.iloc[0])
        return {"found": True, "products": [row], "evidence": [f"products.csv:{row['sku']}"]}

    cols = _name_columns(df) or [c for c in df.columns if c != "sku"]
    text = df[cols].agg(" ".join, axis=1).str.lower()
    ql = q.lower()
    scores = []
    q_tokens = set(re.findall(r"\w+", ql))
    for idx, t in text.items():
        t_tokens = set(re.findall(r"\w+", t))
        overlap = len(q_tokens & t_tokens) / max(1, len(q_tokens))
        ratio = difflib.SequenceMatcher(None, ql, t).ratio()
        scores.append((overlap * 0.7 + ratio * 0.3, idx))
    scores.sort(reverse=True)
    top = [i for s, i in scores[:3] if s > 0.25]
    if not top:
        return {"found": False, "message": "No product matched. Ask the customer for the SKU or exact product name."}
    rows = [_row(df.loc[i]) for i in top]
    return {"found": True, "products": rows,
            "note": "Closest name matches; confirm with the customer if unsure.",
            "evidence": [f"products.csv:{r['sku']}" for r in rows]}


_STOP = {"the", "and", "for", "with", "pack", "set", "of", "inch", "new", "pro", "plus", "mini", "max"}


def find_products_in_text(text: str, max_hits: int = 2) -> list:
    """Catalog products whose name the customer said (e.g. 'Pagecraft A5 Ruled Notebook')."""
    df = products()
    cols = _name_columns(df)
    if not cols or not text:
        return []
    t_tokens = set(re.findall(r"[a-z0-9]+", text.lower()))
    scored = []
    for idx, row in df.iterrows():
        name = " ".join(str(row[c]) for c in cols)
        toks = [w for w in dict.fromkeys(re.findall(r"[a-z0-9]+", name.lower()))
                if len(w) >= 3 and w not in _STOP and not w.isdigit()]
        if not toks:
            continue
        matched = [w for w in toks if w in t_tokens]
        frac = len(matched) / len(toks)
        if len(matched) >= 2 and frac >= 0.4:
            scored.append((frac, len(matched), idx))
    scored.sort(reverse=True)
    return [_row(df.loc[i]) for _, _, i in scored[:max_hits]]


# ---------------------------------------------------------------- pincodes
@lru_cache(maxsize=1)
def pincodes() -> pd.DataFrame:
    return _csv("pincodes.csv")


def check_pincode(pincode: str) -> dict:
    pin = re.sub(r"\D", "", str(pincode or ""))
    hit = pincodes()[pincodes()["pincode"] == pin]
    if not len(hit):
        return {"found": False, "pincode": pin,
                "message": "Pincode not in ZentroMart's serviceable list."}
    return {"found": True, **_row(hit.iloc[0]), "evidence": [f"pincodes.csv:{pin}"]}


# ---------------------------------------------------------------- dates
@lru_cache(maxsize=1)
def holidays() -> set:
    try:
        return {datetime.strptime(d, "%Y-%m-%d").date() for d in _csv("holidays.csv")["date"]}
    except Exception:  # fall back to KB-00.3 table
        return {date(2026, 8, 15), date(2026, 10, 2), date(2026, 10, 19), date(2026, 11, 9)}


def is_business_day(d: date) -> bool:
    return d.weekday() != 6 and d not in holidays()  # Mon-Sat, minus warehouse holidays


def add_business_days(start: date, n: int) -> date:
    d, step = start, (1 if n >= 0 else -1)
    remaining = abs(n)
    while remaining:
        d += timedelta(days=step)
        if is_business_day(d):
            remaining -= 1
    return d


def business_days_between(a: date, b: date) -> int:
    """Business days in (a, b]; negative if b < a."""
    if b < a:
        return -business_days_between(b, a)
    return sum(1 for i in range(1, (b - a).days + 1) if is_business_day(a + timedelta(days=i)))


def _parse_date(s) -> date:
    s = str(s).strip()[:10]
    for fmt in ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            pass
    raise ValueError(f"Unrecognised date {s!r}; use YYYY-MM-DD")


def _fmt(d: date) -> str:
    return f"{d.strftime('%A')}, {d.day} {d.strftime('%B %Y')} ({d.isoformat()})"


def date_calc(start_date: str = None, add_business_days_n: int = None,
              add_calendar_days_n: int = None) -> dict:
    """Deterministic date maths. Today is fixed at 20 Oct 2026."""
    try:
        start = _parse_date(start_date) if start_date else TODAY
    except ValueError as e:
        return {"error": str(e)}
    out = {"today": _fmt(TODAY), "start": _fmt(start),
           "calendar_days_from_start_to_today": (TODAY - start).days,
           "business_days_from_start_to_today": business_days_between(start, TODAY),
           "evidence": ["KB-00.2", "KB-00.3"]}
    if add_business_days_n is not None:
        end = add_business_days(start, int(add_business_days_n))
        out["after_business_days"] = _fmt(end)
        out["is_after_today"] = end > TODAY
    if add_calendar_days_n is not None:
        end = start + timedelta(days=int(add_calendar_days_n))
        out["after_calendar_days"] = _fmt(end)
        out["is_after_today"] = end > TODAY
    return out
