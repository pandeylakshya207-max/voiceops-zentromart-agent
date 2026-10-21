"""Make customer audio from the practice set with Gnani TTS, to test run.py end to end.

  python make_test_audio.py --out test_audio --questions 4 --conversation PC002

Single questions become q_<ID>.wav (one call each); a conversation becomes <ID>_1.wav, <ID>_2.wav …
(played as one call by run.py). A short silent clip (unclear_1.wav) checks recovery from bad audio.
The customer voice is a different Gnani voice from the agent's so the two are easy to tell apart.
"""
import argparse
import wave
from pathlib import Path

import pandas as pd

from src.config import PRACTICE_DIR
from src.voice import Voice

CUSTOMER_VOICE = "Pranav"
LANG = {"en": "en-IN", "hi-en": "hi-en", "hi": "hi-IN"}


def say(voice, text, path, lang):
    try:
        voice.tts.synthesize(text, voice=CUSTOMER_VOICE, model="timbre-v2.5",
                             language=LANG.get(lang, "en-IN"), output_file=str(path))
    except Exception:
        voice.tts.synthesize(text, voice=CUSTOMER_VOICE, model="timbre-v2.5",
                             language="en-IN", output_file=str(path))
    print("wrote", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="test_audio")
    ap.add_argument("--questions", type=int, default=4, help="how many practice questions (mixed languages)")
    ap.add_argument("--conversation", default="PC002")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    voice = Voice()

    q = pd.read_csv(PRACTICE_DIR / "questions.csv", dtype=str, keep_default_na=False)
    picked = pd.concat([q[q["language"] == lang].head(max(1, args.questions // 3)) for lang in ("en", "hi-en", "hi")])
    for _, r in picked.head(args.questions).iterrows():
        say(voice, r["question"], out / f"q_{r['id']}.wav", r["language"])

    c = pd.read_csv(PRACTICE_DIR / "conversations.csv", dtype=str, keep_default_na=False)
    conv = c[c["conversation_id"] == args.conversation].assign(t=lambda d: d["turn"].astype(int)).sort_values("t")
    for _, r in conv.iterrows():
        say(voice, r["customer_message"], out / f"{args.conversation}_{r['turn']}.wav", r["language"])

    with wave.open(str(out / "unclear_1.wav"), "wb") as w:  # 2 s of silence
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(b"\x00\x00" * 32000)
    print("wrote", out / "unclear_1.wav")


if __name__ == "__main__":
    main()
