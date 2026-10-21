"""Chat wrapper over several providers with model fallback, rate-limit backoff and tool-call retry.

Models are listed in LLM_MODELS as "provider:model" (no prefix = groq), e.g.
  openai/gpt-oss-120b, gemini:gemini-2.5-flash, openai/gpt-oss-20b
When a model hits its daily limit (or is unavailable) the next one is used.

  python -m src.llm          # ping every configured model and list your Gemini models
"""
import json
import os
import re
import time
from types import SimpleNamespace

from .config import LLM_MODELS, LLM_TEMPERATURE

GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


class LLMError(RuntimeError):
    pass


def _split(entry: str):
    if ":" in entry and entry.split(":", 1)[0] in ("groq", "gemini"):
        return tuple(entry.split(":", 1))
    return "groq", entry


class LLM:
    def __init__(self, models=None, verbose: bool = False):
        entries = [_split(m) for m in (models or LLM_MODELS)]
        keys = {"groq": os.getenv("GROQ_API_KEY"), "gemini": os.getenv("GEMINI_API_KEY")}
        self.entries = [(p, m) for p, m in entries if keys.get(p)]
        if not self.entries:
            raise LLMError("No usable model: set GROQ_API_KEY (and optionally GEMINI_API_KEY) in .env")
        self._clients = {}
        self.idx = 0
        self.verbose = verbose
        self.calls = 0
        self.tokens = 0
        self.last_finish = None

    # ------------------------------------------------------------------ plumbing
    @property
    def models(self):
        return [m if p == "groq" else f"{p}:{m}" for p, m in self.entries]

    @property
    def model(self):
        return self.models[self.idx]

    def _client(self, provider):
        if provider not in self._clients:
            if provider == "groq":
                from groq import Groq
                self._clients[provider] = Groq(max_retries=1, timeout=60)
            else:
                from openai import OpenAI
                self._clients[provider] = OpenAI(api_key=os.getenv("GEMINI_API_KEY"),
                                                 base_url=GEMINI_BASE_URL, max_retries=0,
                                                 timeout=float(os.getenv("GEMINI_TIMEOUT", "180")))
        return self._clients[provider]

    def _next_model(self, why: str):
        if self.idx + 1 >= len(self.entries):
            raise LLMError(f"All models exhausted. Last error: {why}")
        print(f"[llm] {self.model} unavailable ({why[:120]}); switching to {self.models[self.idx + 1]}")
        self.idx += 1

    def _kwargs(self, provider, model, messages, tools, tool_choice, json_mode, max_tokens, lite):
        kw = dict(model=model, messages=messages, temperature=LLM_TEMPERATURE)
        if tools:
            kw["tools"] = tools
            kw["tool_choice"] = tool_choice or "auto"
        if json_mode:
            kw["response_format"] = {"type": "json_object"}
        if provider == "groq":
            kw["max_completion_tokens"] = max_tokens
            if model.startswith("openai/gpt-oss"):
                kw["extra_body"] = {"reasoning_effort": "low"}
        else:
            kw["max_tokens"] = max_tokens + 1000  # Gemini's thinking tokens count against this
            effort = os.getenv("GEMINI_REASONING", "")  # e.g. "low"; empty = provider default
            if effort and not lite:
                kw["reasoning_effort"] = effort
        return kw

    # ------------------------------------------------------------------ main call
    def chat(self, messages, tools=None, tool_choice=None, json_mode=False, max_tokens=700):
        """Return the assistant message object (has .content and .tool_calls)."""
        attempt, lite = 0, False
        while True:
            attempt += 1
            provider, model = self.entries[self.idx]
            kwargs = self._kwargs(provider, model, messages, tools, tool_choice, json_mode, max_tokens, lite)
            t0 = time.time()
            try:
                resp = self._client(provider).chat.completions.create(**kwargs)
                if self.verbose:
                    print(f"[llm] {self.model} answered in {time.time() - t0:.1f}s")
                self.calls += 1
                self._daily_waits = 0
                if getattr(resp, "usage", None):
                    self.tokens += resp.usage.total_tokens or 0
                self.last_finish = resp.choices[0].finish_reason
                return resp.choices[0].message
            except LLMError:
                raise
            except Exception as e:  # groq.* and openai.* errors both carry status_code
                status = getattr(e, "status_code", None)
                msg = str(e)
                if status is None and re.search(r"timed? ?out|timeout|connection", msg + type(e).__name__, re.I):
                    if attempt <= 4:  # network blips hit every model alike: wait and retry the same one
                        wait_s = 10 * attempt
                        print(f"[llm] network problem ({type(e).__name__}); retrying in {wait_s}s")
                        time.sleep(wait_s)
                        continue
                    self._next_model(f"no response ({type(e).__name__})")
                    attempt, lite = 0, False
                    continue
                if self.verbose:
                    print(f"[llm] {self.model} error {status}: {msg[:300]}")
                if status == 429:
                    wait = re.search(r"(?:try again|retry) in (?:(\d+)m)?([\d.]+)s", msg, re.I)
                    secs = (int(wait.group(1) or 0) * 60 + float(wait.group(2))) if wait else 15
                    if re.search(r"per ?day|TPD|RPD|daily|PerDay", msg, re.I):
                        if self.idx + 1 < len(self.entries):
                            self._next_model("daily limit reached")
                            attempt, lite = 0, False
                            continue
                        self._daily_waits = getattr(self, "_daily_waits", 0) + 1
                        if secs <= 900 and self._daily_waits <= 3:  # last model: window may free up soon
                            print(f"[llm] daily token limit on {self.model}; waiting {secs / 60:.1f} min for it to free up")
                            time.sleep(secs + 2)
                            continue
                        raise LLMError(f"Daily limit reached on all models (next free in {secs / 60:.0f} min)")
                    if secs > 90 or attempt > 8:
                        self._next_model(f"rate limited for {secs:.0f}s")
                        attempt, lite = 0, False
                        continue
                    print(f"[llm] rate limited, waiting {secs:.1f}s")
                    time.sleep(secs + 0.5)
                    continue
                if status in (401, 403, 404) or "model_not_found" in msg or "decommissioned" in msg \
                        or "not found" in msg.lower():
                    self._next_model(f"model not available ({status})")
                    attempt, lite = 0, False
                    continue
                if status == 400 and "tool_use_failed" in msg:
                    if attempt <= 2:
                        continue  # malformed tool call from the model: just retry
                    body = getattr(e, "body", None) or {}
                    err = body.get("error", body) if isinstance(body, dict) else {}
                    gen = err.get("failed_generation") if isinstance(err, dict) else None
                    return SimpleNamespace(content=gen or "", tool_calls=None)
                if status == 400 and provider == "gemini" and not lite:
                    lite = True  # drop optional params (reasoning_effort) and retry
                    continue
                if status == 400 and attempt <= 2 and json_mode:
                    json_mode = False  # some models reject json mode; retry without
                    continue
                if status in (500, 502, 503, 504) and attempt <= 3:
                    time.sleep(3 * attempt)
                    continue
                raise

    def json(self, messages, max_tokens=500) -> dict:
        msg = self.chat(messages, json_mode=True, max_tokens=max_tokens)
        return parse_json(msg.content or "")


def parse_json(text: str) -> dict:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                pass
    return {"reply": text}


if __name__ == "__main__":
    print("Configured chain:", ", ".join(LLM_MODELS))
    if os.getenv("GEMINI_API_KEY"):
        try:
            from openai import OpenAI
            names = [m.id for m in OpenAI(api_key=os.getenv("GEMINI_API_KEY"), base_url=GEMINI_BASE_URL)
                     .models.list()]
            flash = sorted(n.replace("models/", "") for n in names if "flash" in n or "gemma" in n)
            print("Gemini models on your key (flash/gemma):", ", ".join(flash) or "(none)")
        except Exception as e:
            print("Could not list Gemini models:", e)
    else:
        print("GEMINI_API_KEY not set: Gemini models in the chain are skipped.")
    for entry in LLM_MODELS:
        try:
            llm = LLM(models=[entry])
        except LLMError:
            print(f"  SKIP {entry}: no API key for this provider")
            continue
        try:
            out = llm.chat([{"role": "user", "content": "Reply with the single word: ready"}], max_tokens=50)
            print(f"  OK   {entry}: {(out.content or '').strip()[:40]!r}")
        except Exception as e:
            print(f"  FAIL {entry}: {str(e)[:150]}")
