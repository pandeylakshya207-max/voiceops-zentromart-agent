# VoiceOps 2026: ZentroMart voice support agent

Capstone for *From Tokens to Transformers* (C&A Club, IIT Guwahati × Gnani.ai).
A customer-support agent for the fictional store **ZentroMart** that answers from company documents,
looks up orders after verifying the caller, holds a multi-turn conversation, hands over to humans
when policy requires, and talks over voice through **Gnani STT/TTS**.

```
customer audio → Gnani STT → normalizer → agent (LLM + hybrid RAG + tools + call memory) → Gnani TTS → spoken reply
```

## What is in each step

| Step | What it adds | Code |
|---|---|---|
| 1 Prompt-only | LLM with a role/limits system prompt, AI disclosure, no documents | `src/prompts.py` (BASELINE_PROMPT), `evaluate.py --step 1` |
| 2 Grounded (RAG) | Section-level chunks of `kb/*.md` (one chunk per `[SECTION-ID]`), hybrid BM25 + multilingual embeddings, document-precedence re-ranking (policy > agent guide > help > FAQ > marketing), superseded docs down-weighted, "I don't know" rule, chunk IDs logged per answer | `src/kb.py`, `src/agent.py` |
| 3 Conversation | Call memory (order ID, verification, tickets), follow-ups, one clarifying question, caller verification with one retry then lock, `lookup_order` / `create_ticket` from the dataset's `tools.py`, product / pincode / business-day tools, refusals and human handover only backed by `created=True` | `src/agent.py`, `src/data.py` |
| 4 Voice | Gnani STT in, Gnani TTS out, replies rewritten for the ear, ASR clean-up of spoken IDs ("ele zero zero one to" → `ELE-0012`, "8978 2837" → `89782837`) | `src/voice.py`, `src/normalize.py`, `run.py` |

## Setup (Windows PowerShell)

```powershell
git clone <this repo> voiceops; cd voiceops
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env      # then put your keys in .env
```

Extract the capstone dataset so this path exists (or set `DATASET_DIR` in `.env`):

```
voiceops/dataset/Capstone_Project_Dataset_FTTT_2026/{kb,data,practice,tools}
```

Keys (`.env`): `GROQ_API_KEY` for the LLM, `GNANI_API_KEY` for speech, optional `GEMINI_API_KEY` as a second LLM provider. Never commit `.env`. `python -m src.llm` pings every configured model.

## Run

**Batch (organizers' command)**: a folder of customer audio files in, replies out:

```powershell
python run.py --audio_dir path\to\audio --out_dir outputs
```

For every file it writes `outputs/<name>.txt` (reply text) and `outputs/<name>.wav` (spoken reply),
plus `outputs/results.jsonl` / `results.csv` with the transcript, decision, tool calls and the
document chunks used. Each file is its own call; files named with a turn marker
(`PC002_1.wav`, `call7_turn2.wav`) are played in order as one call. `--independent` forces one call per file.

**Text chat**: `python chat.py --debug`

**Evaluation on the practice set**

```powershell
python evaluate.py --step 1   # prompt-only baseline  -> results/step1_questions_asr_text.csv
python evaluate.py --step 2   # RAG agent             -> results/step2_questions_asr_text.csv
python evaluate.py --step 3   # conversations         -> results/step3_conversations.csv
```

Scoring follows the practice files: every `;`-group of `must_include` must appear (`/` = alternatives),
nothing from `must_not_include`, decision within `acceptable_decisions`, expected tool calls made
with matching arguments, and `gold_evidence` recall over retrieved + cited chunks.
Summaries accumulate in `results/summary.json`.

## Design notes

- **Grounding first.** Every turn retrieves the top sections before the LLM speaks; the model can
  search again with `search_kb`. Facts from data come only from tools (orders, products, pincodes),
  and all date arithmetic goes through `date_calc` (today fixed at 20 Oct 2026; business days Mon–Sat
  minus warehouse holidays; return windows in calendar days).
- **Never claim what the code didn't do.** A handover decision without a successful ticket triggers
  a real `create_ticket` call; the reply only mentions a ticket when one exists.
- **Privacy.** Order details are only shown after the phone last-4 or delivery pincode matches; two
  failures lock the order for the rest of the call.
- **Rate limits.** `LLM_MODELS` is a fallback chain across Groq and Gemini models; the client switches model on
  daily limits and waits on per-minute limits.
