"""Step 4: Gnani STT (ears) and TTS (voice) around the same agent."""
import re
import time
from pathlib import Path

from .agent import reply_language
from .config import STT_LANGUAGE, TTS_MODEL, TTS_VOICE

TTS_LANG = {"en": "en-IN", "hi-en": "hi-en", "hi": "hi-IN"}
DIDNT_CATCH = {
    "en": "Sorry, I didn't catch that. Could you say it again?",
    "hi-en": "Sorry, main sun nahi paayi. Kya aap dobara bol sakte hain?",
    "hi": "माफ़ कीजिए, मैं सुन नहीं पाई। क्या आप दोबारा बोल सकते हैं?",
}
FILLERS = {"uh", "um", "umm", "hmm", "hm", "ah", "er", "hello", "hi", "haan", "ok", "okay", "yes", "no",
           "ji", "accha", "acha", "huh"}


def is_unclear(heard: str) -> bool:
    """Empty, noise-only or a lone filler word: ask the caller to repeat."""
    words = re.findall(r"[^\W_]+", (heard or "").lower())
    if not words:
        return True
    return len(words) <= 2 and all(w in FILLERS for w in words) and not re.search(r"\d", heard)


def speech_text(reply: str) -> str:
    """Rewrite a reply for the ear (Gnani text-normalization guidance)."""
    t = reply
    t = re.sub(r"\b(\d{4})(\d{4})\b", r"\1 \2", t)  # 8-digit order IDs -> '8978 2837'
    t = re.sub(r"\b([A-Z]{2,5})-(\d+)\b", lambda m: f"{m.group(1)} {' '.join(m.group(2))}", t)  # SKU spelled out
    t = re.sub(r"₹\s?([\d,]+(?:\.\d+)?)", r"\1 rupees", t)
    t = t.replace("&", " and ")
    return re.sub(r"\s+", " ", t).strip()


def padded_copy(audio_path, seconds: float, attempt: int = 1):
    """Same speech, different bytes: a little silence in front and the volume scaled by a few percent
    (inaudible), written under a neutral name. Used when the STT firewall rejects a file. None if not PCM WAV."""
    import tempfile
    import wave

    import numpy as np
    try:
        with wave.open(str(audio_path), "rb") as r:
            params, frames = r.getparams(), r.readframes(r.getnframes())
        if params.sampwidth != 2:
            return None
        samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) * (0.97 - 0.03 * attempt)
        samples = np.clip(samples + np.random.default_rng(attempt).normal(0, 2, samples.shape), -32768, 32767)
        pad = np.zeros(int(params.framerate * seconds) * params.nchannels, dtype=np.float32)
        data = np.concatenate([pad, samples]).astype("<i2").tobytes()
        out = Path(tempfile.gettempdir()) / f"caller_audio_{attempt}.wav"
        with wave.open(str(out), "wb") as w:
            w.setnchannels(params.nchannels)
            w.setsampwidth(2)
            w.setframerate(params.framerate)
            w.writeframes(data)
        return str(out)
    except Exception:
        return None


class Voice:
    def __init__(self):
        from gnani.stt import GnaniSTTClient
        from gnani.tts import GnaniTTSClient

        self.stt = GnaniSTTClient()  # reads GNANI_API_KEY from the environment
        self.tts = GnaniTTSClient()

    def transcribe(self, audio_path, language_code: str = STT_LANGUAGE, attempts: int = 3) -> str:
        """Gnani STT with retries: transient 403/429/5xx or network errors are retried with a short wait."""
        path = str(audio_path)
        for i in range(attempts):
            try:
                out = self.stt.transcribe(path, language_code=language_code)
                break
            except Exception as e:
                if i == attempts - 1:
                    raise
                blocked = "403" in str(e)
                if blocked:  # the STT firewall rejected this exact file: resend the same speech with new bytes
                    path = padded_copy(audio_path, 0.3 * (i + 1), i + 1) or path
                wait = 2 if blocked else 3 * (i + 1)
                print(f"[voice] STT attempt {i + 1} failed ({str(e)[:40]}...); "
                      f"{'re-encoding audio and ' if blocked else ''}retrying in {wait}s")
                time.sleep(wait)
        if isinstance(out, dict):
            return (out.get("transcript") or "").strip()
        return str(out or "").strip()

    def speak(self, text: str, output_file, lang: str = "en") -> str:
        output_file = str(output_file)
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        language = TTS_LANG.get(lang, "en-IN")
        try:
            self.tts.synthesize(speech_text(text), voice=TTS_VOICE, model=TTS_MODEL,
                                language=language, output_file=output_file)
        except Exception:
            if language == "en-IN":
                raise
            # voice/language combination not supported: fall back to Indian English
            self.tts.synthesize(speech_text(text), voice=TTS_VOICE, model=TTS_MODEL,
                                language="en-IN", output_file=output_file)
        return output_file

    def voice_turn(self, audio_path, agent, output_file="reply.wav"):
        """The brief's voice_turn: audio in -> transcript -> agent -> spoken reply."""
        try:
            heard = self.transcribe(audio_path)
        except Exception as e:  # unreadable / too long / network: recover instead of crashing the call
            print(f"[voice] STT failed: {e.__class__.__name__}: {e}")
            heard = ""
        if is_unclear(heard) and not (agent.state.turns == 0 and heard.strip()):
            lang = agent.state.lang or "en"
            reply = DIDNT_CATCH.get(lang, DIDNT_CATCH["en"])
            if agent.state.turns == 0:
                reply = "Hi, I'm ZentroMart's AI assistant. " + reply
            turn = {"user": heard, "reply": reply, "decision": "clarify", "tool_calls": [],
                    "retrieved": [], "evidence": [], "chunks": [], "unclear_audio": True}
        else:
            turn = agent.turn(heard)
            reply = turn["reply"]
        self.speak(reply, output_file, reply_language(reply))
        return heard, turn
