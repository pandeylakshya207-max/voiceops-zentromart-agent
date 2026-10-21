"""Central configuration: paths, models, reference date."""
import os
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Dataset location (the extracted capstone ZIP). Override with DATASET_DIR in .env.
DATASET_DIR = Path(os.getenv(
    "DATASET_DIR", ROOT / "dataset" / "Capstone_Project_Dataset_FTTT_2026"))
KB_DIR = DATASET_DIR / "kb"
DATA_DIR = DATASET_DIR / "data"
PRACTICE_DIR = DATASET_DIR / "practice"
TOOLS_FILE = DATASET_DIR / "tools" / "tools.py"

CACHE_DIR = ROOT / ".cache"
RESULTS_DIR = ROOT / "results"

# LLM: comma-separated fallback chain ("provider:model", no prefix = groq).
# On a daily limit / unavailable model the agent moves to the next one.
LLM_MODELS = [m.strip() for m in os.getenv(
    "LLM_MODELS",
    "openai/gpt-oss-120b,openai/gpt-oss-20b",
).split(",") if m.strip()]
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))

# Retrieval
EMBED_MODEL = os.getenv("EMBED_MODEL", "intfloat/multilingual-e5-small")
TOP_K = int(os.getenv("TOP_K", "6"))

# Voice (Gnani)
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "en-IN")  # en-IN also accepts Hindi-English mix
TTS_MODEL = os.getenv("TTS_MODEL", "timbre-v2.5")
TTS_VOICE = os.getenv("TTS_VOICE", "Kaveri")

# The dataset is frozen on this date (KB-00.2).
TODAY = date(2026, 10, 20)
