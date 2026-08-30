"""CrewAI research crew: fetch live financial rates and prove where each came from.

Search is done in Python, up front and in parallel, then handed to the crew as a
pre-built evidence pack. Two agents then reason over it: an extractor pulls out
structured rates, and an adversarial verifier deletes anything it cannot tie
back to a quoted source.

Why search is not left to agent tool-calling: retrieving a fixed list of known
queries is plumbing, not judgement. Doing it in Python collapses many sequential
agent turns into two LLM calls plus a handful of parallel HTTP requests, which is
faster, cheaper, reproducible, and survives a 5-requests-per-minute free tier.
The Serper tool is still registered on the extractor so it *can* search for
anything the pack failed to cover.

Python then re-checks the verifier's work in `validate_rates`, because an agent
claiming it verified something is not the same as it being verified.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import requests
from crewai import LLM, Agent, Crew, Process, Task
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

from config import (
    CACHE_DIR,
    CREW_MAX_RPM,
    CREW_MODEL,
    RATES,
    RESEARCH_CACHE_HOURS,
    SERPER_API_KEY,
)
from schemas import RateResearchResult, ValidatedRate

SERPER_URL = "https://google.serper.dev/search"
CACHE_FILE = CACHE_DIR / "rates.json"
RESULTS_PER_QUERY = 6


# --- Search ----------------------------------------------------------------
def serper_search(query: str) -> str:
    """One Serper query, rendered as source-attributed snippets.

    Every result is prefixed with its URL so the model has something concrete to
    cite. A failure returns an explicit marker rather than an empty string, so
    the agent is told the lookup failed instead of inferring there is no data.
    """
    try:
        response = requests.post(
            SERPER_URL,
            headers={"X-API-KEY": SERPER_API_KEY, "Content-Type": "application/json"},
            json={"q": query, "gl": "in", "hl": "en", "num": 8},
            timeout=25,
        )
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
    except (requests.RequestException, ValueError) as exc:
        return f"SEARCH_FAILED: {exc}. Do not guess a number; omit this rate."

    lines: list[str] = []
    box = payload.get("answerBox") or {}
    if box:
        answer = box.get("answer") or box.get("snippet") or ""
        if answer:
            lines.append(f"SOURCE: {box.get('link', '')}\nSNIPPET: {answer}")

    for item in payload.get("organic", [])[:RESULTS_PER_QUERY]:
        snippet = (item.get("snippet") or "").strip()
        if not snippet:
            continue
        dated = f" (published {item['date']})" if item.get("date") else ""
        lines.append(
            f"SOURCE: {item.get('link', '')}{dated}\nSNIPPET: {snippet}"
        )

    return "\n\n".join(lines) if lines else "NO_RESULTS. Omit this rate."


def fetch_evidence(keys: list[str]) -> dict[str, str]:
    """Run every rate's search concurrently. Pure I/O, no LLM involved."""
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(keys)))) as pool:
        results = pool.map(lambda k: (k, serper_search(RATES[k].query)), keys)
    return dict(results)


def format_evidence_pack(evidence: dict[str, str]) -> str:
    return "\n\n".join(
        f"===== RATE KEY: {key} ({RATES[key].label}) =====\n"
        f"Search used: \"{RATES[key].query}\"\n\n{block}"
        for key, block in evidence.items()
    )


# --- Tool ------------------------------------------------------------------
class SearchInput(BaseModel):
    query: str = Field(description="A focused web search query")


class SerperSearchTool(BaseTool):
    """Escape hatch for the extractor when the pre-built pack is inadequate.

    Hand-written rather than pulled from `crewai-tools` to keep the dependency
    tree small and to control exactly what the agent sees.
    """

    name: str = "web_search"
    description: str = (
        "Search the live web for current financial rates in India. Use ONLY if the "
        "evidence pack lacks a usable figure for a rate. Returns results with a "
        "SOURCE url and a SNIPPET; quote snippets verbatim and never invent a number."
    )
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str) -> str:
        return serper_search(query)


# --- Crew ------------------------------------------------------------------
def build_crew(keys: list[str], evidence_pack: str) -> Crew:
    llm = LLM(model=CREW_MODEL, temperature=0.0)

    wanted = "\n".join(f"- {k} ({RATES[k].label})" for k in keys)

    extractor = Agent(
        role="Indian Financial Rates Analyst",
        goal="Extract each requested rate from the evidence pack and cite the exact source sentence.",
        backstory=(
            "You are a meticulous financial data analyst. You have been burned before by "
            "publishing a number you could not source, so you now refuse to report any "
            "figure that is not visible verbatim in the evidence in front of you."
        ),
        tools=[SerperSearchTool()],
        llm=llm,
        max_iter=8,
        verbose=False,
        allow_delegation=False,
    )

    verifier = Agent(
        role="Data Verification Officer",
        goal="Delete every rate that is not provably supported by its quoted evidence.",
        backstory=(
            "You are an auditor, not an assistant. You are rewarded for removing "
            "unsupported figures and penalised for letting one through. You never add, "
            "adjust or repair numbers."
        ),
        llm=llm,
        max_iter=5,
        verbose=False,
        allow_delegation=False,
    )

    extract_task = Task(
        description=(
            "Below is a pack of live web search results, grouped by rate key.\n\n"
            f"{evidence_pack}\n\n"
            f"Extract the current value for each of these rates:\n{wanted}\n\n"
            "Rules you must not break:\n"
            "1. Every rate needs a source_url and an `evidence` field containing the "
            "VERBATIM snippet sentence in which the number appeared. Copy it exactly.\n"
            "2. The number in `value_pct` must literally appear in `evidence`.\n"
            "3. Report annual percentages as plain numbers: 7.1 means 7.1%.\n"
            "4. If several banks are listed, report the best mainstream value and quote "
            "the snippet showing it.\n"
            "5. If the pack has no credible number for a rate, OMIT that rate entirely. "
            "An omission is correct behaviour; a guess is a failure. Use web_search only "
            "if you genuinely need one more lookup.\n"
            "6. Never compute, average or adjust a number yourself. Report what the "
            "source says.\n"
            "7. Match the instrument exactly: a fixed deposit snippet is not a PPF rate, "
            "and a 1-year return is not a 10-year CAGR."
        ),
        expected_output=(
            "A list of findings, each with: key, label, value_pct, as_of, source_url, evidence."
        ),
        agent=extractor,
    )

    verify_task = Task(
        description=(
            "Audit the analyst's findings. For each rate, check that:\n"
            "- the numeric value in value_pct actually appears inside the evidence text;\n"
            "- the evidence is about the right instrument and the right time period;\n"
            "- source_url is a real URL;\n"
            "- the value is plausible for that instrument.\n\n"
            "DELETE any finding that fails. Keep the survivors byte-for-byte identical. "
            "Do not repair, re-estimate or invent replacements. Returning fewer rates is "
            "the correct outcome when the evidence is weak."
        ),
        expected_output="The filtered, evidence-backed list of rates as structured data.",
        agent=verifier,
        context=[extract_task],
        output_pydantic=RateResearchResult,
    )

    return Crew(
        agents=[extractor, verifier],
        tasks=[extract_task, verify_task],
        process=Process.sequential,
        max_rpm=CREW_MAX_RPM,
        verbose=False,
    )


# --- Deterministic validation ---------------------------------------------
_NUM = re.compile(r"\d+(?:\.\d+)?")


def _number_supported_by_evidence(value: float, evidence: str) -> bool:
    """Check the cited snippet really contains the claimed number.

    This is the guardrail that does not rely on the model's honesty: we parse
    every number out of the evidence text and require a near-exact match.
    """
    if not evidence:
        return False
    for match in _NUM.findall(evidence.replace(",", "")):
        try:
            if abs(float(match) - value) <= 0.051:
                return True
        except ValueError:
            continue
    return False


def validate_rates(
    result: RateResearchResult, keys: list[str]
) -> tuple[dict[str, ValidatedRate], list[str]]:
    """Accept only rates that are in range and genuinely cited; fall back for the rest.

    Returns the rate book plus the list of keys that had to fall back, which the
    graph uses to decide whether a research retry is worthwhile.
    """
    accepted: dict[str, ValidatedRate] = {}
    rejections: dict[str, str] = {}

    for found in result.rates:
        key = found.key.strip().lower()
        if key not in RATES or key in accepted:
            continue
        spec = RATES[key]
        if not (spec.low <= found.value_pct <= spec.high):
            rejections[key] = (
                f"{found.value_pct}% is outside the plausible {spec.low}-{spec.high}% band"
            )
            continue
        if not found.source_url.startswith("http"):
            rejections[key] = "no usable source URL"
            continue
        if not _number_supported_by_evidence(found.value_pct, found.evidence):
            rejections[key] = f"{found.value_pct} does not appear in the quoted evidence"
            continue
        accepted[key] = ValidatedRate(
            key=key,
            label=spec.label,
            value_pct=found.value_pct,
            origin="researched",
            as_of=found.as_of,
            source_url=found.source_url,
            evidence=found.evidence.strip(),
        )

    failed: list[str] = []
    for key in keys:
        if key in accepted:
            continue
        failed.append(key)
        accepted[key] = ValidatedRate(
            key=key,
            label=RATES[key].label,
            value_pct=RATES[key].fallback,
            origin="fallback",
            as_of="static assumption",
            reject_reason=rejections.get(key, "not returned by research"),
        )

    return accepted, failed


# --- Cache -----------------------------------------------------------------
def load_cached_rates() -> dict[str, ValidatedRate] | None:
    if not CACHE_FILE.exists():
        return None
    try:
        blob: dict[str, Any] = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        if time.time() - blob["fetched_at"] > RESEARCH_CACHE_HOURS * 3600:
            return None
        return {k: ValidatedRate(**v) for k, v in blob["rates"].items()}
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def save_cached_rates(rates: dict[str, ValidatedRate]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(
        json.dumps(
            {"fetched_at": time.time(), "rates": {k: v.model_dump() for k, v in rates.items()}},
            indent=2,
        ),
        encoding="utf-8",
    )


def _is_gemini_exhausted(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "prepayment credits are depleted" in text or "billing#prepay" in text


def _fallback_book(keys: list[str], reason: str) -> tuple[dict[str, ValidatedRate], list[str]]:
    rates, failed = validate_rates(RateResearchResult(), keys)
    for key in failed:
        rates[key].reject_reason = reason
    return rates, failed


_GEMINI_UNAVAILABLE = False


def research_rates(keys: list[str] | None = None) -> tuple[dict[str, ValidatedRate], list[str]]:
    """Search, extract, verify and validate. Always returns a complete rate book.

    Never raises: an API outage or exhausted quota degrades to labelled fallbacks
    with a stated reason, because a partial plan the user can audit beats a
    stack trace.
    """
    global _GEMINI_UNAVAILABLE
    keys = keys or list(RATES)
    if _GEMINI_UNAVAILABLE:
        return _fallback_book(
            keys, "Gemini billing/quota exhausted; labelled assumptions used"
        )
    try:
        pack = format_evidence_pack(fetch_evidence(keys))
        output = build_crew(keys, pack).kickoff()
        try:
            parsed = getattr(output, "pydantic", None)
        except Exception as exc:  # noqa: BLE001 - .pydantic can trigger another model call
            raise RuntimeError(str(exc)) from exc
        result = parsed if isinstance(parsed, RateResearchResult) else None
        if result is None or not isinstance(result, RateResearchResult):
            try:
                result = RateResearchResult.model_validate_json(str(output))
            except Exception:  # noqa: BLE001 - malformed model output is expected
                result = RateResearchResult()
        return validate_rates(result, keys)
    except Exception as exc:  # noqa: BLE001 - external APIs fail; the plan must not
        if _is_gemini_exhausted(exc):
            _GEMINI_UNAVAILABLE = True
            reason = "Gemini billing/quota exhausted; labelled assumptions used"
        else:
            reason = f"research unavailable ({type(exc).__name__})"
        return _fallback_book(keys, reason)
