"""Web demo: talk to the ZentroMart agent in the browser (mic or typing), hear the reply.

  python app.py            # then open http://127.0.0.1:7860

Each browser tab is its own phone call; "New call" starts over. The panel on the right shows what
Gnani STT heard, the agent's decision, tool calls and the document sections it used.
"""
import tempfile
import uuid
from pathlib import Path

import gradio as gr

from src.agent import Agent, reply_language
from src.kb import Retriever
from src.llm import LLM

LLM_SHARED = LLM()
KB_SHARED = Retriever()
VOICE = None
OUT_DIR = Path(tempfile.gettempdir()) / "zentromart_replies"
OUT_DIR.mkdir(exist_ok=True)


def _voice():
    global VOICE
    if VOICE is None:
        from src.voice import Voice
        VOICE = Voice()
    return VOICE


def _new_agent():
    return Agent(LLM_SHARED, KB_SHARED)


def _details(turn, heard=None):
    lines = []
    if heard is not None:
        lines.append(f"**Heard (Gnani STT):** {heard or '(nothing)'}")
    lines.append(f"**Decision:** `{turn.get('decision')}`")
    for c in turn.get("tool_calls", []):
        lines.append(f"**Tool:** `{c['name']}` {c['args']} → {c['result']}")
    chunks = turn.get("chunks") or []
    if chunks:
        lines.append("**Sources used:**")
        for c in chunks[:4]:
            lines.append(f"- `{c['id']}`: {c['text'][:220]}…")
    return "\n\n".join(lines)


def _speak(reply):
    path = OUT_DIR / f"reply_{uuid.uuid4().hex[:8]}.wav"
    try:
        _voice().speak(reply, path, reply_language(reply))
        return str(path)
    except Exception as e:
        print(f"[app] TTS failed: {e}")
        return None


def on_voice(audio_path, history, agent):
    agent = agent or _new_agent()
    if not audio_path:
        return history, None, "Record something first.", agent, None
    out = OUT_DIR / f"reply_{uuid.uuid4().hex[:8]}.wav"
    heard, turn = _voice().voice_turn(audio_path, agent, out)
    history = history + [{"role": "user", "content": heard or "(unclear audio)"},
                         {"role": "assistant", "content": turn["reply"]}]
    return history, str(out), _details(turn, heard), agent, None


def on_text(text, history, agent, speak):
    agent = agent or _new_agent()
    if not text.strip():
        return history, None, "", agent, ""
    turn = agent.turn(text)
    history = history + [{"role": "user", "content": text}, {"role": "assistant", "content": turn["reply"]}]
    audio = _speak(turn["reply"]) if speak else None
    return history, audio, _details(turn), agent, ""


def on_reset():
    return [], None, "New call started.", _new_agent(), None, ""


with gr.Blocks(title="ZentroMart voice support") as demo:
    gr.Markdown("## ZentroMart support: AI voice agent\nSpeak or type like a customer. "
                "Speech in and out uses Gnani STT/TTS; answers come from ZentroMart's documents and tools.")
    agent_state = gr.State(None)
    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.Chatbot(type="messages", height=420, label="Call")
            reply_audio = gr.Audio(label="Agent's spoken reply", autoplay=True, interactive=False)
            with gr.Row():
                mic = gr.Audio(sources=["microphone", "upload"], type="filepath", label="Your voice")
                send_voice = gr.Button("Send voice", variant="primary")
            with gr.Row():
                text = gr.Textbox(placeholder="…or type here and press Enter", show_label=False, scale=4)
                speak = gr.Checkbox(value=True, label="Speak reply", scale=1)
            new_call = gr.Button("New call")
        with gr.Column(scale=2):
            details = gr.Markdown("Details of each turn appear here.")

    send_voice.click(on_voice, [mic, chat, agent_state], [chat, reply_audio, details, agent_state, mic])
    text.submit(on_text, [text, chat, agent_state, speak], [chat, reply_audio, details, agent_state, text])
    new_call.click(on_reset, None, [chat, reply_audio, details, agent_state, mic, text])

if __name__ == "__main__":
    demo.launch()
