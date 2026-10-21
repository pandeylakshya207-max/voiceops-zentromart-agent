"""Evaluate the agent on the practice set.

  python evaluate.py --step 1            # Step 1: prompt-only baseline on practice/questions.csv
  python evaluate.py --step 2            # Step 2: RAG + tools agent on practice/questions.csv
  python evaluate.py --step 3            # Step 3: multi-turn agent on practice/conversations.csv
  options: --field question|asr_text (default asr_text), --limit N, --ids PQ0002,PQ0003, --debug,
           --resume (keep results already saved for this step and run only the missing items),
           --redo PQ0040,PQ0044 (run those again and replace their saved results)
Results are saved after every item, so a stop (quota, Ctrl+C) never loses finished work.
"""
import argparse
import json
import time

import pandas as pd

from src.config import PRACTICE_DIR, RESULTS_DIR
from src.llm import LLM, LLMError, parse_json
from src.prompts import BASELINE_PROMPT
from src.scoring import score_row


def _load(name, ids=None, limit=None, key="id"):
    df = pd.read_csv(PRACTICE_DIR / name, dtype=str, keep_default_na=False)
    if ids:
        df = df[df[key].isin(ids)]
    if limit:
        keep = df[key].drop_duplicates().head(limit)
        df = df[df[key].isin(keep)]
    return df


def step1(llm, df, field, rows=None):
    rows = rows if rows is not None else []
    for _, r in df.iterrows():
        q = r[field] or r["question"]
        try:
            out = parse_json(llm.chat([{"role": "system", "content": BASELINE_PROMPT},
                                       {"role": "user", "content": q}], json_mode=True, max_tokens=400).content)
        except LLMError as e:
            print(f"\n[stopped early] {e}\nSaving the results so far (rerun with --resume to continue).")
            break
        reply, decision = str(out.get("reply", "")), str(out.get("decision", ""))
        s = score_row(reply, decision, r.to_dict())
        rows.append({"id": r["id"], "intent": r["intent"], "input": q, "reply": reply,
                     "decision": decision, "expected": r["acceptable_decisions"] or r["expected_decision"], **s})
        print(f"{'✓' if s['correct'] else '✗'} {r['id']}  {decision:9s} {reply[:90]}")
        _save(rows)
    return rows


DEBUG = False
CHECKPOINT = None  # set in main(): path to this step's CSV
BOOL_COLS = ("include_ok", "exclude_ok", "decision_ok", "correct", "tools_ok")


def _save(rows):
    if CHECKPOINT and rows:
        RESULTS_DIR.mkdir(exist_ok=True)
        pd.DataFrame(rows).to_csv(CHECKPOINT, index=False, encoding="utf-8-sig")


def _load_previous(path):
    if not path.exists():
        return []
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    rows = df.to_dict("records")
    for r in rows:
        for c in BOOL_COLS:
            if c in r:
                r[c] = str(r[c]) == "True"
        for c in ("evidence_recall", "cited_recall"):
            if c in r:
                r[c] = float(r[c]) if r[c] not in ("", "None", "nan") else None
    return rows


def _debug(t):
    if DEBUG:
        print(f"      input: {t.get('user', '')[:200]!r}")
        print(f"      raw: {t.get('raw_final', '')[:600]!r}")
        for c in t["tool_calls"]:
            print(f"      tool {c['name']}({c['args']}) -> {str(c['result'])[:200]}")
        print(f"      retrieved: {t['retrieved']}  evidence: {t['evidence']}")


def step2(agent, df, field, rows=None):
    rows = rows if rows is not None else []
    for _, r in df.iterrows():
        q = r[field] or r["question"]
        agent.reset()
        try:
            t = agent.turn(q)
        except LLMError as e:
            print(f"\n[stopped early] {e}\nSaving the results so far (rerun with --resume to continue).")
            break
        found = t["retrieved"] + t["evidence"]
        s = score_row(t["reply"], t["decision"], r.to_dict(), found_evidence=found)
        s["cited_recall"] = score_row(t["reply"], t["decision"], r.to_dict(), found_evidence=t["evidence"])["evidence_recall"]
        rows.append({"id": r["id"], "intent": r["intent"], "input": q, "reply": t["reply"],
                     "decision": t["decision"], "expected": r["acceptable_decisions"] or r["expected_decision"],
                     "gold_evidence": r["gold_evidence"], "retrieved": ";".join(t["retrieved"]),
                     "evidence": ";".join(t["evidence"]),
                     "tools": ";".join(c["name"] for c in t["tool_calls"]), "model": agent.llm.model, **s})
        _save(rows)
        print(f"{'✓' if s['correct'] else '✗'} {r['id']}  {t['decision']:9s} R={s['evidence_recall']}  {t['reply'][:90]}")
        if not s["correct"]:
            why = [x for x in (s["missing"] and f"missing: {s['missing']}",
                               s["forbidden_found"] and f"forbidden: {s['forbidden_found']}",
                               (not s["decision_ok"]) and f"decision not in {r['acceptable_decisions']}") if x]
            print(f"      ↳ {' | '.join(why)}")
            _debug(t)
    return rows


def step3(agent, df, rows=None):
    rows = rows if rows is not None else []
    df = df.assign(_turn=df["turn"].astype(int)).sort_values(["conversation_id", "_turn"])
    for cid, conv in df.groupby("conversation_id", sort=False):
        agent.reset()
        print(f"--- {cid} ({conv['scenario'].iloc[0]})")
        stop, conv_rows = False, []
        for _, r in conv.iterrows():
            try:
                t = agent.turn(r["customer_message"])
            except LLMError as e:
                print(f"\n[stopped early] {e}\nSaving the results so far (rerun with --resume to continue).")
                stop = True
                break
            found = t["retrieved"] + t["evidence"]
            s = score_row(t["reply"], t["decision"], r.to_dict(), found_evidence=found, tool_calls=t["tool_calls"])
            conv_rows.append({"conversation_id": cid, "turn": r["turn"], "scenario": r["scenario"],
                         "customer": r["customer_message"], "reply": t["reply"], "decision": t["decision"],
                         "expected": r["expected_decision"], "expected_tools": r["expected_tool_calls"],
                         "tools": json.dumps([{"name": c["name"], "args": c["args"]} for c in t["tool_calls"]],
                                             ensure_ascii=False),
                         "gold_evidence": r["gold_evidence"], "retrieved": ";".join(t["retrieved"]),
                         "model": agent.llm.model, **s})
            print(f"  {'✓' if s['correct'] else '✗'} t{r['turn']} {t['decision']:9s} {t['reply'][:100]}")
            if not s["correct"]:
                why = [x for x in (s["missing"] and f"missing: {s['missing']}",
                                   s["forbidden_found"] and f"forbidden: {s['forbidden_found']}",
                                   (not s["decision_ok"]) and f"decision≠{r['expected_decision']}",
                                   s.get("tools_missing") and f"tools: {s['tools_missing']}") if x]
                print(f"      ↳ {' | '.join(why)}")
                _debug(t)
        if stop:
            print(f"(conversation {cid} was cut off and is not saved; --resume will rerun it)")
            break
        rows.extend(conv_rows)  # only complete conversations are kept
        _save(rows)
    return rows


def summarize(rows, step, llm):
    df = pd.DataFrame(rows)
    summ = {"step": step, "n": len(df), "accuracy": round(float(df["correct"].mean()), 4),
            "include_ok": round(float(df["include_ok"].mean()), 4),
            "exclude_ok": round(float(df["exclude_ok"].mean()), 4),
            "decision_ok": round(float(df["decision_ok"].mean()), 4),
            "model_calls": llm.calls, "tokens": llm.tokens,
            "models": (df["model"].value_counts().to_dict() if "model" in df else llm.models[: llm.idx + 1])}
    if "evidence_recall" in df and df["evidence_recall"].notna().any():
        summ["evidence_recall"] = round(float(df["evidence_recall"].dropna().mean()), 4)
    if "tools_ok" in df:
        summ["tools_ok"] = round(float(df["tools_ok"].mean()), 4)
    return df, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", type=int, choices=[1, 2, 3], required=True)
    ap.add_argument("--field", default="asr_text", choices=["asr_text", "question"])
    ap.add_argument("--limit", type=int)
    ap.add_argument("--ids", help="comma-separated question or conversation IDs")
    ap.add_argument("--debug", action="store_true", help="print raw model output and tool calls for failures")
    ap.add_argument("--resume", action="store_true", help="skip items already saved for this step")
    ap.add_argument("--redo", help="comma-separated IDs to run again and replace in the saved results")
    args = ap.parse_args()
    global DEBUG
    DEBUG = args.debug
    ids = args.ids.split(",") if args.ids else None

    global CHECKPOINT
    name = {1: f"step1_questions_{args.field}", 2: f"step2_questions_{args.field}",
            3: "step3_conversations"}[args.step]
    CHECKPOINT = RESULTS_DIR / f"{name}.csv"
    key = "conversation_id" if args.step == 3 else "id"
    prev = _load_previous(CHECKPOINT) if (args.resume or args.redo) else []
    if args.redo:
        redo = set(args.redo.split(","))
        prev = [r for r in prev if r[key] not in redo]
    done = {r[key] for r in prev}
    data = _load("conversations.csv" if args.step == 3 else "questions.csv", ids, args.limit, key=key)
    todo = data[~data[key].isin(done)]
    if args.resume or args.redo:
        print(f"Keeping {len(done)} saved results, running {todo[key].nunique()}")

    llm = LLM()
    t0 = time.time()
    try:
        if args.step == 1:
            rows = step1(llm, todo, args.field, prev)
        else:
            from src.agent import Agent
            agent = Agent(llm)
            rows = (step2(agent, todo, args.field, prev) if args.step == 2 else step3(agent, todo, prev))
    except KeyboardInterrupt:
        print("\n[interrupted] results so far are saved; rerun with --resume to continue")
        rows = _load_previous(CHECKPOINT) if CHECKPOINT.exists() else prev

    if not rows:
        print("No results to save (no model available). Try again later.")
        return
    df, summ = summarize(rows, args.step, llm)
    summ["seconds"] = round(time.time() - t0, 1)
    summ["complete"] = bool(len({r[key] for r in rows}) >= data[key].nunique())
    _save(rows)
    allp = RESULTS_DIR / "summary.json"
    allsum = json.loads(allp.read_text()) if allp.exists() else {}
    allsum[name] = summ
    allp.write_text(json.dumps(allsum, indent=2))
    print("\n" + json.dumps(summ, indent=2))
    print(f"Saved results/{name}.csv")


if __name__ == "__main__":
    main()
