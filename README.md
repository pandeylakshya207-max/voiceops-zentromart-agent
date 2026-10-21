# VoiceOps 2026: ZentroMart voice support agent

Capstone for *From Tokens to Transformers* (C&A Club, IIT Guwahati × Gnani.ai).
A customer-support agent for the fictional store **ZentroMart** that answers from company documents,
looks up orders after verifying the caller, holds a multi-turn conversation, hands over to humans
when policy requires, and talks over voice through **Gnani STT/TTS**.

```
customer audio → Gnani STT → normalizer → agent (LLM + hybrid RAG + tools + call memory) → Gnani TTS → spoken reply
```

**Repository:** https://github.com/pandeylakshya207-max/voiceops-zentromart-agent

## Quick start for evaluators

```powershell
git clone https://github.com/pandeylakshya207-max/voiceops-zentromart-agent.git voiceops
cd voiceops
python -m venv .venv; .\.venv\Scripts\Activate.ps1      # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                                   # Linux/macOS: cp .env.example .env ; then add your keys
# put the capstone dataset at dataset/Capstone_Project_Dataset_FTTT_2026 (or set DATASET_DIR in .env)
python run.py --audio_dir path\to\customer_audio --out_dir outputs
```

## Results on the practice set

| | Step 1: prompt only | Step 2: RAG + tools |
|---|---|---|
| Questions fully correct (52) | 17.3% | **98.1%** (51/52) |
| Required facts present (`must_include`) | 26.9% | 100% |
| Trap answers avoided (`must_not_include`) | 100% | 100% |
| Correct decision | 76.9% | 98.1% |
| Gold evidence retrieved | n/a | 87.9% |

Step 3 (23 practice conversations, 96 turns): **100%** of turns fully correct, 100% expected tool calls. Details, failures and fixes are in [report.pdf](report.pdf).

Same 52 questions, scored on their ASR-style `asr_text`. Step 2 rows: Groq gpt-oss-120b 25, gpt-oss-20b 21, Gemini 3.5 Flash Lite 6 (free-tier limits).
Full per-question results are in `results/`.

## What is in each step

| Step | What it adds | Code |
|---|---|---|
| 1 Prompt-only | LLM with a role/limits system prompt, AI disclosure, no documents | `src/prompts.py` (BASELINE_PROMPT), `evaluate.py --step 1` |
| 2 Grounded (RAG) | Section-level chunks of `kb/*.md` (one chunk per `[SECTION-ID]`), hybrid BM25 + multilingual embeddings, document-precedence re-ranking (policy > agent guide > help > FAQ > marketing), superseded docs down-weighted, "I don't know" rule, chunk IDs logged per answer | `src/kb.py`, `src/agent.py` |
| 3 Conversation | Call memory (order ID, verification, tickets), follow-ups, one clarifying question, caller verification with one retry then lock, `lookup_order` / `create_ticket` from the dataset's `tools.py`, product / pincode / business-day tools, refusals and human handover only backed by `created=True` | `src/agent.py`, `src/data.py` |
| 4 Voice | Gnani STT in, Gnani TTS out, replies rewritten for the ear, ASR clean-up of spoken IDs ("ele zero zero one to" → `ELE-0012`, "8978 2837" → `89782837`) | `src/voice.py`, `src/normalize.py`, `run.py` |

## Setup (Windows PowerShell)

```powershell
git clone https://github.com/pandeylakshya207-max/voiceops-zentromart-agent.git voiceops; cd voiceops
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

**Make test audio** from the practice set with Gnani TTS (customer voice "Pranav"), then run the batch:

```powershell
python make_test_audio.py --out test_audio --questions 4 --conversation PC002
python run.py --audio_dir test_audio --out_dir sample_outputs
```

`sample_outputs/` in this repo was produced exactly this way.

**Web demo (voice or text)**: `python app.py`, then open http://127.0.0.1:7860. Record with the mic or
type; the reply is spoken with Gnani TTS and the side panel shows what STT heard, the decision, tool calls
and the document sections used. "New call" starts a fresh conversation.

**Text chat in the terminal**: `python chat.py --debug`

**Evaluation on the practice set**

```powershell
python evaluate.py --step 1   # prompt-only baseline  -> results/step1_questions_asr_text.csv
python evaluate.py --step 2   # RAG agent             -> results/step2_questions_asr_text.csv
python evaluate.py --step 3   # conversations         -> results/step3_conversations.csv
```

Scoring follows the practice files: every `;`-group of `must_include` must appear (`/` = alternatives),
nothing from `must_not_include`, decision within `acceptable_decisions`, expected tool calls made
with matching arguments, and `gold_evidence` recall over retrieved + cited chunks.
Summaries accumulate in `results/summary.json`. Results are saved after every item:
`--resume` continues an interrupted run (e.g. after a free-tier daily limit) and
`--redo PQ0040,PQ0044` re-runs chosen items and replaces their results. `--debug` prints the raw model
output, tool calls and retrieved sections for every failure.

## Design notes

- **Grounding first.** Every turn retrieves the top sections before the LLM speaks; the model can
  search again with `search_kb`. Facts from data come only from tools (orders, products, pincodes),
  and all date arithmetic goes through `date_calc` (today fixed at 20 Oct 2026; business days Mon–Sat
  minus warehouse holidays; return windows in calendar days).
- **Never claim what the code didn't do.** A handover decision without a successful ticket triggers
  a real `create_ticket` call; the reply only mentions a ticket when one exists.
- **Privacy.** Order details are only shown after the phone last-4 or delivery pincode matches; two
  failures lock the order for the rest of the call.
- **Verifier loop.** Before a reply is spoken, code checks it and sends the model back once when it
  (a) states a date no tool or document gave, or ignores the date `date_calc` computed, (b) says "I don't
  know" without having searched, (c) asks for an order ID or other detail it does not need, (d) answers in
  the wrong language, (e) leaves out the deadline for a return/claim on a verified order, (f) leaves out a
  catalog fact that was asked (e.g. warranty months), or (g) handles a damage or account-security report
  without checking whether a human team must take it.
- **Deterministic steps in code.** Order lookup, caller verification (phone last 4 / pincode, one retry,
  then lock), product lookup for SKUs or product names the caller says, and escalation tickets run in
  code, so required tool calls cannot be skipped by the model.
- **Recovery from unclear audio.** Empty or filler-only transcripts ("uh", "hello?") get a short
  "sorry, could you say that again?" in the caller's language; an order number or verification value
  heard with the wrong number of digits is read back ("I heard 8978 283, which is 7 digits…") and the
  caller is asked to repeat, without spending an LLM call.
- **Rate limits.** `LLM_MODELS` is a fallback chain (Groq models, optionally Gemini); the client waits on
  per-minute limits, retries network errors, and moves to the next model on daily limits.
