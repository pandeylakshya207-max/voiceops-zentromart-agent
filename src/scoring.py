"""Scoring rules mirroring the practice files.

must_include:      groups separated by ';' are ALL required; alternatives inside a group are separated by '/'.
must_not_include:  items separated by ';' (or '/'); any one present = fail.
acceptable_decisions / expected_decision: ';'-separated list.
gold_evidence:     ';'-separated section IDs or 'file.csv:key' rows.
expected_tool_calls: 'name(arg=val, ...)' items separated by ';'.
"""
import re
import unicodedata


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").lower()
    s = s.replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", s)


def _split(s: str, sep: str):
    return [p.strip() for p in (s or "").split(sep) if p.strip()]


def _contains(text: str, needle: str) -> bool:
    t, n = _norm(text), _norm(needle)
    if re.fullmatch(r"[a-z0-9 ]+", n) and len(n) <= 4:  # short ascii tokens: whole-word match
        return re.search(rf"(?<![a-z0-9]){re.escape(n)}(?![a-z0-9])", t) is not None
    return n in t


def check_must_include(text: str, spec: str):
    missing = []
    for group in _split(spec, ";"):
        alts = _split(group, "/")
        if alts and not any(_contains(text, a) for a in alts):
            missing.append(group)
    return not missing, missing


def check_must_not_include(text: str, spec: str):
    bad = [x for x in re.split(r"[;/]", spec or "") if x.strip() and _contains(text, x.strip())]
    return not bad, bad


def check_decision(decision: str, acceptable: str) -> bool:
    ok = _split(acceptable, ";")
    return not ok or (decision or "") in ok


def evidence_recall(found: list, gold: str):
    gold_items = _split(gold, ";")
    if not gold_items:
        return None
    found_set = {f.strip().lower() for f in found}
    hit = [g for g in gold_items if g.lower() in found_set]
    return len(hit) / len(gold_items)


_CALL = re.compile(r"(\w+)\s*\(([^)]*)\)")


def parse_tool_calls(spec: str):
    calls = []
    for name, args in _CALL.findall(spec or ""):
        kv = {}
        for part in args.split(","):
            if "=" in part:
                k, v = part.split("=", 1)
                kv[k.strip()] = v.strip().strip("'\"")
        calls.append((name, kv))
    return calls


def _digits(v):
    d = re.sub(r"\D", "", str(v or ""))
    return d or str(v or "").strip().lower()


def check_tool_calls(made: list, spec: str):
    """Each expected call must match a made call with the same name and the listed args."""
    expected = parse_tool_calls(spec)
    missing = []
    for name, kv in expected:
        ok = any(m["name"] == name and all(_digits(m["args"].get(k)) == _digits(v) for k, v in kv.items())
                 for m in made)
        if not ok:
            missing.append(f"{name}({', '.join(f'{k}={v}' for k, v in kv.items())})")
    return not missing, missing


def score_row(reply: str, decision: str, row: dict, found_evidence=None, tool_calls=None) -> dict:
    inc_ok, missing = check_must_include(reply, row.get("must_include", ""))
    exc_ok, bad = check_must_not_include(reply, row.get("must_not_include", ""))
    acceptable = row.get("acceptable_decisions") or row.get("expected_decision", "")
    dec_ok = check_decision(decision, acceptable)
    res = {"include_ok": inc_ok, "missing": "; ".join(missing), "exclude_ok": exc_ok,
           "forbidden_found": "; ".join(bad), "decision_ok": dec_ok}
    if tool_calls is not None:
        tc_ok, tc_missing = check_tool_calls(tool_calls, row.get("expected_tool_calls", ""))
        res.update(tools_ok=tc_ok, tools_missing="; ".join(tc_missing))
    else:
        tc_ok = True
    res["evidence_recall"] = evidence_recall(found_evidence or [], row.get("gold_evidence", "")) \
        if found_evidence is not None else None
    res["correct"] = inc_ok and exc_ok and dec_ok and tc_ok
    return res
