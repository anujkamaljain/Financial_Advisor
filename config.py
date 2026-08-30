"""Central configuration: env keys, model names, and the rate table."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import NamedTuple

from dotenv import load_dotenv

load_dotenv()

# CrewAI's event bus logs emoji status glyphs. On a Windows console defaulting to
# cp1252 that raises UnicodeEncodeError mid-run, so widen the streams before any
# agent code gets a chance to log. Must happen at import time.
if sys.platform == "win32":
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
CACHE_DIR = ROOT / ".cache"

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
SERPER_API_KEY = os.getenv("SERPER_API_KEY", "")

# CrewAI needs the "gemini/" provider prefix; LangChain does not. Note that Google
# retires specific model versions for new API keys, so if you get a 404 saying a
# model "is no longer available to new users", set a current one in .env.
CREW_MODEL = os.getenv("CREW_MODEL", "gemini/gemini-3.6-flash")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gemini-3.6-flash")
EMBED_MODEL = os.getenv("EMBED_MODEL", "models/gemini-embedding-001")

# Live rates are cached for a day so repeated runs are fast, cheap and reproducible.
RESEARCH_CACHE_HOURS = float(os.getenv("RESEARCH_CACHE_HOURS", "24"))
MAX_RESEARCH_RETRIES = int(os.getenv("MAX_RESEARCH_RETRIES", "1"))

# The Gemini free tier allows as few as 5 requests per minute. Throttling the crew
# makes it wait instead of dying, which matters because a 429 mid-run would
# otherwise discard work already paid for.
CREW_MAX_RPM = int(os.getenv("CREW_MAX_RPM", "4"))

# CrewAI otherwise opens an interactive "view your traces? [y/N]" prompt on first
# run, which hangs a non-interactive script. Must be set before crewai imports.
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("CREWAI_TELEMETRY_OPT_OUT", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

# The SDKs log transport-level chatter (automatic-function-calling advice, retry
# notices) that is noise in a chat UI. Retries are handled and reported as notes,
# so only genuine failures need to reach the console.
for _noisy in ("google_genai", "google.genai", "httpx", "opentelemetry", "chromadb"):
    logging.getLogger(_noisy).setLevel(logging.ERROR)

class Rate(NamedTuple):
    """One researchable rate: how to find it, what counts as sane, what to assume.

    `low`/`high` form the plausibility band -- the guardrail that makes a
    hallucinated number unusable. They are deliberately wide, so they catch
    nonsense (a "45% PPF rate") rather than genuine market movement.
    `fallback` is used only when research fails validation, so the app degrades
    to a labelled assumption instead of inventing a figure.
    """

    label: str
    query: str
    low: float
    high: float
    fallback: float


# Every rate here is consumed by the allocation engine. Rates that were merely
# interesting (EPF, 1-year FD, NPS) were removed: each one costs a web search and
# prompt space, and EPF in particular is deducted before salary is credited, so it
# is never allocated out of take-home pay.
RATES: dict[str, Rate] = {
    "ppf": Rate(
        "Public Provident Fund",
        "current PPF interest rate India latest quarter",
        5.0, 10.0, 7.1,
    ),
    "bank_savings": Rate(
        "Savings / liquid fund",
        "highest savings account interest rate India banks latest",
        2.0, 8.0, 3.0,
    ),
    "debt_fund": Rate(
        "Short-duration debt fund",
        "debt mutual fund 3 year average returns India latest",
        4.0, 10.0, 7.0,
    ),
    "nifty50_10y_cagr": Rate(
        "Nifty 50 (10-year CAGR)",
        "Nifty 50 index 10 year CAGR returns latest",
        6.0, 20.0, 12.0,
    ),
    "gold_10y_cagr": Rate(
        "Gold (10-year CAGR)",
        "gold 10 year annualised return India latest",
        2.0, 20.0, 11.0,
    ),
    "inflation_cpi": Rate(
        "CPI inflation",
        "India CPI retail inflation rate latest month",
        1.0, 12.0, 5.0,
    ),
}


def require_keys() -> None:
    """Fail loudly and early rather than mid-pipeline with a confusing traceback."""
    missing = [
        name
        for name, value in (("GEMINI_API_KEY", GEMINI_API_KEY), ("SERPER_API_KEY", SERPER_API_KEY))
        if not value
    ]
    if missing:
        raise SystemExit(
            f"Missing required environment variable(s): {', '.join(missing)}.\n"
            "Copy .env.example to .env and fill in your keys."
        )
    # CrewAI's Gemini provider reads the key from the environment. Only GEMINI_API_KEY
    # is set: google-genai warns and picks a winner if GOOGLE_API_KEY is also present.
    # The LangChain clients are passed the key explicitly instead.
    os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
