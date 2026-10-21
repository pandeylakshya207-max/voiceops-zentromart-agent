"""Clean up ASR transcripts: spoken digits -> numerals, spoken SKUs and order IDs -> canonical form.

Example: "so sku ele-zero zero one to pe zentromart assured hai kya"
      -> "so sku ELE-0012 pe zentromart assured hai kya"   entities: sku=ELE-0012
"""
import re

# Unambiguous number words
DIGITS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9",
    # Hindi / Hinglish
    "shunya": "0", "sunya": "0", "teen": "3", "chaar": "4", "char": "4", "paanch": "5",
    "panch": "5", "chhe": "6", "chhah": "6", "che": "6", "saat": "7", "aath": "8", "nau": "9",
}
# Words that are digits only when next to spelled-out digits ("one to" -> 12, but "want to" stays)
AMBIGUOUS = {"o": "0", "oh": "0", "to": "2", "too": "2", "for": "4", "ate": "8",
             "won": "1", "ek": "1", "do": "2", "free": "3", "tree": "3"}
MULTIPLIERS = {"double": 2, "triple": 3}
TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16,
         "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
        "eighty": 80, "ninety": 90}
SCALES = {"hundred": 100, "thousand": 1000, "lakh": 100000, "lac": 100000}
AMBIGUOUS["and"] = ""

_TOKEN = re.compile(r"[A-Za-zऀ-ॿ]+|\d+|[^\sA-Za-z\dऀ-ॿ]")


def _spoken_kind(tok: str):
    t = tok.lower()
    if t in DIGITS or t in TEENS or t in TENS or t in SCALES:
        return "word"
    if t in AMBIGUOUS:
        return "amb"
    if t in MULTIPLIERS:
        return "mult"
    if tok.isdigit():
        return "num"
    return None


def spoken_digits_to_numerals(text: str) -> str:
    tokens = _TOKEN.findall(text)
    kinds = [_spoken_kind(t) for t in tokens]
    n = len(tokens)

    # An ambiguous word counts only if a spelled-out digit (or multiplier) is adjacent,
    # possibly through other ambiguous words.
    keep = [k in ("word", "num", "mult") for k in kinds]
    changed = True
    while changed:
        changed = False
        for i, k in enumerate(kinds):
            if k == "amb" and not keep[i]:
                nb = [j for j in (i - 1, i + 1) if 0 <= j < n]
                if any(keep[j] and kinds[j] in ("word", "mult", "amb") for j in nb):
                    keep[i] = changed = True

    out, i = [], 0
    while i < n:
        if keep[i] and kinds[i] != "num":
            run, j = [], i
            while j < n and keep[j]:
                run.append(j)
                j += 1
            # need at least one real spelled digit in the run to convert
            if any(kinds[r] == "word" for r in run) or (len(run) > 1 and any(kinds[r] == "mult" for r in run)):
                out.append(_run_to_digits([tokens[r] for r in run], [kinds[r] for r in run]))
            else:
                out.extend(tokens[r] for r in run)
            i = j
        else:
            out.append(tokens[i])
            i += 1
    # re-join, keeping hyphens tight
    text2 = " ".join(out)
    text2 = re.sub(r"\s*-\s*", "-", text2)
    text2 = re.sub(r"\s+([,.?!])", r"\1", text2)
    return text2


def _unit(t: str, kind: str):
    return int(t) if kind == "num" else int(DIGITS.get(t, AMBIGUOUS.get(t, "0")) or 0)


def _run_to_digits(toks, kinds) -> str:
    """'eight nine seven eight' -> 8978 · 'forty five thirty three' -> 4533 ·
    'four thousand five hundred thirty three' -> 4533 · 'double five' -> 55."""
    low = [t.lower() for t in toks]
    if any(t in SCALES for t in low):  # a cardinal number
        total, cur = 0, 0
        for t, k in zip(low, kinds):
            if t in SCALES:
                if SCALES[t] == 100:
                    cur = max(cur, 1) * 100
                else:
                    total += max(cur, 1) * SCALES[t]
                    cur = 0
            elif t in TENS:
                cur += TENS[t]
            elif t in TEENS:
                cur += TEENS[t]
            elif k in ("word", "amb", "num"):
                cur += _unit(t, k)
        return str(total + cur)
    out, mult, i = "", 1, 0
    while i < len(low):  # spoken digit by digit or in pairs
        t, k = low[i], kinds[i]
        if k == "mult":
            mult = MULTIPLIERS[t]
        elif t in TENS:
            val = TENS[t]
            if i + 1 < len(low) and low[i + 1] in DIGITS and DIGITS[low[i + 1]] != "0":
                val += int(DIGITS[low[i + 1]])
                i += 1
            out += str(val)
        elif t in TEENS:
            out += str(TEENS[t])
        elif k == "num":
            out += toks[i] * mult
            mult = 1
        else:
            d = DIGITS.get(t, AMBIGUOUS.get(t, ""))
            out += d * mult
            mult = 1
        i += 1
    return out


def _digit_groups(text: str):
    """Runs of digit groups separated only by spaces, e.g. '8978 2837' -> ['8978', '2837']."""
    return [m.group(0) for m in re.finditer(r"\d+(?:[ ]+\d+)*", text)]


def normalize(text: str, sku_prefixes=None) -> dict:
    """Return {'text': normalized_text, 'entities': {...}} for an ASR or typed utterance."""
    raw = text or ""
    t = spoken_digits_to_numerals(raw)

    entities = {"order_ids": [], "skus": [], "pincodes": [], "four_digit": []}

    # SKUs like ELE-0012 / "ele 0012" / "ele-00 12"
    prefixes = sorted(sku_prefixes or [], key=len, reverse=True)
    if prefixes:
        pat = re.compile(r"\b(" + "|".join(map(re.escape, prefixes)) + r")[\s-]*(\d(?:\s?\d){2,5})\b", re.I)

        def _sku(m):
            digits = re.sub(r"\s", "", m.group(2))
            sku = m.group(1).upper() + "-" + digits
            entities["skus"].append(sku)
            return sku
        t = pat.sub(_sku, t)
    for m in re.finditer(r"\b([A-Z]{2,5})-(\d{3,6})\b", t):
        s = f"{m.group(1)}-{m.group(2)}"
        if s not in entities["skus"]:
            entities["skus"].append(s)

    # Numbers: join space-separated groups (spoken in chunks) and classify by length
    sku_digits = {s.split("-")[1] for s in entities["skus"]}
    for grp in _digit_groups(t):
        joined = grp.replace(" ", "")
        parts = grp.split()
        if joined in sku_digits:
            continue
        if len(joined) == 8:
            entities["order_ids"].append(joined)
        elif len(joined) == 6:
            entities["pincodes"].append(joined)
        elif len(joined) == 4:
            entities["four_digit"].append(joined)
        else:
            for p in parts:
                if len(p) == 8:
                    entities["order_ids"].append(p)
                elif len(p) == 6:
                    entities["pincodes"].append(p)
                elif len(p) == 4:
                    entities["four_digit"].append(p)
    entities = {k: list(dict.fromkeys(v)) for k, v in entities.items() if v}
    return {"text": t, "entities": entities}
