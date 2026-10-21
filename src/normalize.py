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

_TOKEN = re.compile(r"[A-Za-zऀ-ॿ]+|\d+|[^\sA-Za-z\dऀ-ॿ]")


def _spoken_kind(tok: str):
    t = tok.lower()
    if t in DIGITS:
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
                digits, mult = "", 1
                for r in run:
                    t = tokens[r].lower()
                    if kinds[r] == "mult":
                        mult = MULTIPLIERS[t]
                        continue
                    d = tokens[r] if kinds[r] == "num" else DIGITS.get(t, AMBIGUOUS.get(t, ""))
                    digits += d * mult
                    mult = 1
                out.append(digits)
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
