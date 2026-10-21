"""Step 4: Gnani STT (ears) and TTS (voice) around the same agent."""
import re
from pathlib import Path

from .agent import reply_language
from .config import STT_LANGUAGE, TTS_MODEL, TTS_VOICE

TTS_LANG = {"en": "en-IN", "hi-en": "hi-en", "hi": "hi-IN"}
DIDNT_CATCH = {
    "en": "Sorry, I didn't catch that. Could you say it again?",
    "hi-en": "Sorry, main sun nahi paayi. Kya aap dobara bol sakte hain?",
}


def speech_text(reply: str) -> str:
    """Rewrite a reply for the ear (Gnani text-normalization guidance)."""
    t = reply
    t = re.sub(r"\b(\d{4})(\d{4})\b", r"\1 \2", t)  # 8-digit order IDs -> '8978 2837'
    t = re.sub(r"\b([A-Z]{2,5})-(\d+)\b", lambda m: f"{m.group(1)} {' '.join(m.group(2))}", t)  # SKU spelled out
    t = re.sub(r"₹\s?([\d,]+(?:\.\d+)?)", r"\1 rupees", t)
    t = t.replace("&", " and ")
    return re.sub(r"\s+", " ", t).strip()


class Voice:
    def __init__(self):
        from gnani.stt import GnaniSTTClient
        from gnani.tts import GnaniTTSClient

        self.stt = GnaniSTTClient()  # reads GNANI_API_KEY from the environment
        self.tts = GnaniTTSClient()

    def transcribe(self, audio_path, language_code: str = STT_LANGUAGE) -> str:
        out = self.stt.transcribe(str(audio_path), language_code=language_code)
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
        heard = self.transcribe(audio_path)
        if not heard:
            reply = DIDNT_CATCH["en"]
            turn = {"user": "", "reply": reply, "decision": "clarify", "tool_calls": [],
                    "retrieved": [], "evidence": [], "chunks": []}
        else:
            turn = agent.turn(heard)
            reply = turn["reply"]
        self.speak(reply, output_file, reply_language(reply))
        return heard, turn
