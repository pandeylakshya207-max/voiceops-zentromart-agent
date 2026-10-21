"""Batch runner for the organizers.

  python run.py --audio_dir path/to/audio --out_dir outputs

Each audio file is treated as a separate call, EXCEPT files whose names mark a turn
(e.g. PC002_1.wav, PC002_2.wav, call7_turn3.mp3, abc_t2.wav), which are played in turn order within
one call so the agent keeps memory. For every file it saves:
  <out_dir>/<name>.txt   the reply text
  <out_dir>/<name>.wav   the spoken reply (Gnani TTS)
and <out_dir>/results.jsonl + results.csv with transcript, reply, decision, tool calls and the
document chunks used.
"""
import argparse
import json
import re
import sys
import time
from collections import OrderedDict
from pathlib import Path

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".webm", ".aac"}
CONV_RE = re.compile(r"^(.+?)[_-](?:turn|t)[_-]?(\d{1,3})$|^(PC\d+)[_-](\d{1,3})$", re.I)


def group_calls(files, independent=False):
    calls = OrderedDict()
    for f in files:
        m = None if independent else CONV_RE.match(f.stem)
        if m:
            key, order = (m.group(1), int(m.group(2))) if m.group(1) else (m.group(3), int(m.group(4)))
        else:
            key, order = f.stem, 0
        calls.setdefault(key, []).append((order, f))
    return {k: [f for _, f in sorted(v)] for k, v in calls.items()}


def main():
    ap = argparse.ArgumentParser(description="Run the ZentroMart voice agent on a folder of audio files")
    ap.add_argument("--audio_dir", required=True)
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--independent", action="store_true", help="treat every file as its own call")
    args = ap.parse_args()

    from src.agent import Agent
    from src.llm import LLM
    from src.voice import Voice

    audio_dir, out_dir = Path(args.audio_dir), Path(args.out_dir)
    files = sorted(p for p in audio_dir.iterdir() if p.suffix.lower() in AUDIO_EXT)
    if not files:
        sys.exit(f"No audio files found in {audio_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)

    agent, voice = Agent(LLM()), Voice()
    calls = group_calls(files, args.independent)
    print(f"{len(files)} audio files -> {len(calls)} calls")

    records = []
    with open(out_dir / "results.jsonl", "w", encoding="utf-8") as jf:
        for call_id, call_files in calls.items():
            agent.reset()
            for i, f in enumerate(call_files, 1):
                t0 = time.time()
                rec = {"file": f.name, "call_id": call_id, "turn": i}
                try:
                    heard, turn = voice.voice_turn(f, agent, out_dir / f"{f.stem}.wav")
                    rec.update(transcript=heard, reply=turn["reply"], decision=turn["decision"],
                               tool_calls=turn["tool_calls"], retrieved=turn["retrieved"],
                               evidence=turn["evidence"], chunks=turn["chunks"],
                               reply_audio=f"{f.stem}.wav")
                except Exception as e:  # keep going: one bad file must not stop the batch
                    rec.update(error=f"{e.__class__.__name__}: {e}", reply="", decision="error")
                rec["seconds"] = round(time.time() - t0, 2)
                (out_dir / f"{f.stem}.txt").write_text(rec.get("reply", ""), encoding="utf-8")
                jf.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
                jf.flush()
                records.append(rec)
                print(f"[{f.name}] heard: {rec.get('transcript', '')[:70]!r}\n    -> {rec.get('decision')}: "
                      f"{rec.get('reply', rec.get('error', ''))[:110]}")

    import pandas as pd
    pd.DataFrame([{**{k: r.get(k) for k in ("file", "call_id", "turn", "transcript", "reply", "decision",
                                              "reply_audio", "seconds", "error")},
                   "tools": ";".join(c["name"] for c in r.get("tool_calls", [])),
                   "chunks_used": ";".join(c["id"] for c in r.get("chunks", []))} for r in records]
                 ).to_csv(out_dir / "results.csv", index=False, encoding="utf-8-sig")
    print(f"\nDone. Replies, audio and sources saved in {out_dir}/")


if __name__ == "__main__":
    main()
