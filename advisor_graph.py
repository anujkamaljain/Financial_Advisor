"""LangGraph orchestration.

    extract_profile ──▶ research ──▶ compute ──▶ advise
           │              ▲   │                    │
           │              └───┘                    ▼
           └──────────────────────────────────▶ (follow-up)  END

The cycle on `research` is the reason this is a graph and not a function call:
when Python-side validation rejects a rate, control flows back for another
attempt at only the failed keys before giving up and using labelled fallbacks.
`extract_profile` also branches straight to `advise` for follow-up questions, so
a chat turn that adds no new facts costs no search quota.
"""

from __future__ import annotations

import re
import time
import warnings
from functools import lru_cache
from typing import Annotated, Literal, TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from config import CHAT_MODEL, GEMINI_API_KEY, MAX_RESEARCH_RETRIES, RATES
from finance import build_plan, compute_tax, format_inr, load_tax_rules
from knowledge import get_kb
from research import _fallback_book, load_cached_rates, research_rates, save_cached_rates
from schemas import (
    FinancialPlan,
    ProfileExtraction,
    TaxBreakdown,
    UserProfile,
    ValidatedRate,
)


class AdvisorState(TypedDict, total=False):
    message: str
    profile: UserProfile | None
    profile_changed: bool
    rates: dict[str, ValidatedRate]
    failed_keys: list[str]
    attempts: int
    plan: FinancialPlan | None
    tax: TaxBreakdown | None
    answer: str
    notes: Annotated[list[str], lambda a, b: (a or []) + (b or [])]


# Newer Gemini models use fixed sampling and warn that temperature is ignored.
# Temperature is still passed so older pinned models behave deterministically.
warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")


@lru_cache(maxsize=4)
def _chat(temperature: float = 0.2) -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=CHAT_MODEL, temperature=temperature, google_api_key=GEMINI_API_KEY
    )


# Gemini's free tier permits as few as 5 requests per minute, so a burst of turns
# will hit 429 in normal use. Retrying is the difference between a working demo
# and a confusing failure.
RETRY_ATTEMPTS = 3
_RETRY_HINT = re.compile(r"retry in (\d+(?:\.\d+)?)s", re.IGNORECASE)


def _is_billing_exhausted(exc: Exception) -> bool:
    """Prepay/billing 429s will not recover by waiting; retrying burns the remaining quota."""
    text = str(exc).lower()
    return "prepayment credits are depleted" in text or "billing#prepay" in text


def _is_rate_limit(exc: Exception) -> bool:
    if _is_billing_exhausted(exc):
        return False
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text or "exceeded your current quota" in text


def _retry_delay(exc: Exception, attempt: int) -> float:
    """Honour the server's own retry hint when it gives one."""
    hint = _RETRY_HINT.search(str(exc))
    if hint:
        return min(float(hint.group(1)) + 1.0, 90.0)
    return min(5.0 * 2**attempt, 60.0)


def _invoke_with_retry(runnable, payload):
    """Invoke a LangChain runnable, backing off on rate limits only.

    Other errors propagate immediately -- retrying a bad request just wastes quota.
    """
    for attempt in range(RETRY_ATTEMPTS):
        try:
            return runnable.invoke(payload)
        except Exception as exc:  # noqa: BLE001 - provider exception types vary
            if _is_billing_exhausted(exc) or not _is_rate_limit(exc) or attempt == RETRY_ATTEMPTS - 1:
                raise
            time.sleep(_retry_delay(exc, attempt))
    raise RuntimeError("unreachable")


def _response_text(response) -> str:
    """Flatten an AIMessage to plain text.

    Gemini returns `content` as a list of typed blocks rather than a string, and
    `.text` is a property in current langchain-core but a method in older ones.
    The string check comes first because the compatibility shim is callable *and*
    a string, and calling it emits a deprecation warning.
    """
    text = getattr(response, "text", None)
    if not isinstance(text, str) and callable(text):
        text = text()
    if isinstance(text, str) and text.strip():
        return text.strip()

    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content.strip()
    parts = [
        block if isinstance(block, str) else str(block.get("text", ""))
        for block in (content or [])
        if isinstance(block, (str, dict))
    ]
    return "\n".join(p for p in parts if p).strip()


# --- Nodes -----------------------------------------------------------------
EXTRACT_PROMPT = """Extract only the financial facts the user explicitly stated.

Rules:
- Convert Indian units to plain rupees: "1 lakh" / "1 lakhs" -> 100000, "1.2 lakhs" -> 120000,
  "1.2L" -> 120000, "1 crore" -> 10000000. Singular and plural unit spellings are the same.
- Set salary_is_annual to true ONLY if the user clearly said per year, per annum, CTC or annual.
  Indian users saying "my salary is X" almost always mean monthly in-hand pay.
- "no EMI" / "no loans" means monthly_emi = 0, not null.
- Leave a field null if the user did not mention it. Never guess or infer a plausible value.
- risk_profile must be one of conservative, moderate, aggressive, or null.

User message: {message}"""


_LAKH = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*(lakhs?|lacs?)\b")
_LPA = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*lpa\b")
_RUPEE_SALARY = re.compile(
    r"(?i)(?:salary|earn|take[\s-]*home|in[\s-]*hand).{0,24}(\d{4,9})"
)
_AGE = re.compile(r"(?i)\b(?:i(?:['’]m| am)|age(?:\s*is)?)\s*(\d{1,2})\b")
_ANNUAL = re.compile(r"(?i)\b(per year|per annum|annual|ctc|lpa)\b")
_MONTHLY = re.compile(r"(?i)\b(per month|/\s*month|monthly)\b")
_NO_EMI = re.compile(r"(?i)\bno\s+(emi|emis|loans?)\b")


def parse_indian_salary(message: str) -> tuple[float | None, bool]:
    """Best-effort rupee amount and whether it is annual. Used when the LLM skips a figure."""
    annual = bool(_ANNUAL.search(message)) and not _MONTHLY.search(message)
    match = _LAKH.search(message) or _LPA.search(message)
    if match:
        return float(match.group(1)) * 100_000, annual or bool(_LPA.search(message))
    match = _RUPEE_SALARY.search(message)
    if match:
        return float(match.group(1)), annual
    return None, False


def parse_stated_age(message: str) -> int | None:
    match = _AGE.search(message)
    if not match:
        return None
    age = int(match.group(1))
    return age if 18 <= age <= 75 else None


def extract_profile(state: AdvisorState) -> AdvisorState:
    """Pull stated facts out of the message and merge them onto any existing profile."""
    previous = state.get("profile")
    notes: list[str] = []
    found = ProfileExtraction()
    parsed_salary, parsed_annual = parse_indian_salary(state["message"])
    if parsed_salary is not None:
        found.salary_amount = parsed_salary
        found.salary_is_annual = parsed_annual
    else:
        try:
            llm = _chat(0.0).with_structured_output(ProfileExtraction)
            found = _invoke_with_retry(
                llm, EXTRACT_PROMPT.format(message=state["message"])
            )
        except Exception as exc:  # noqa: BLE001 - still parse remaining fields below
            notes.append(
                f"Could not read new details from that message ({type(exc).__name__})."
            )
            found = ProfileExtraction()
    if found.age is None:
        found.age = parse_stated_age(state["message"])
    if found.monthly_emi is None and _NO_EMI.search(state["message"]):
        found.monthly_emi = 0.0

    monthly: float | None = None
    if found.salary_amount:
        amount = found.salary_amount
        # Deterministic guard: the model occasionally passes "1.2" through from
        # "1.2 lakhs" instead of expanding it. No real salary figure is < 1000.
        if amount < 1000:
            amount *= 100_000
            notes.append(f"Read the salary figure as {format_inr(amount)}.")
        monthly = amount / 12 if found.salary_is_annual else amount
        notes.append(
            f"Assuming {format_inr(monthly)} is your monthly take-home pay"
            + (" (derived from the annual figure you gave)." if found.salary_is_annual
               else ". Say 'annual' or 'CTC' if you meant something else.")
        )

    updates = {
        k: v
        for k, v in {
            "age": found.age,
            "risk_profile": found.risk_profile,
            "dependents": found.dependents,
            "monthly_essentials": found.monthly_essentials,
            "monthly_emi": found.monthly_emi,
            "emergency_fund_now": found.emergency_fund_now,
            "has_term_insurance": found.has_term_insurance,
            "has_health_insurance": found.has_health_insurance,
        }.items()
        if v is not None
    }

    if previous is None and monthly is None:
        if notes and notes[0].startswith("Could not read new details"):
            notes.append("Please try again in a minute.")
        return {"profile": None, "profile_changed": False, "notes": notes}

    if previous is None:
        profile = UserProfile(monthly_take_home=monthly, **updates)  # type: ignore[arg-type]
        changed = True
    else:
        merged = previous.model_dump()
        if monthly is not None:
            merged["monthly_take_home"] = monthly
        merged.update(updates)
        profile = UserProfile(**merged)
        changed = profile.model_dump() != previous.model_dump()

    return {"profile": profile, "profile_changed": changed, "notes": notes}


def research(state: AdvisorState) -> AdvisorState:
    """Fetch live rates via the CrewAI crew, reusing the daily cache when valid."""
    attempts = state.get("attempts", 0)

    if attempts == 0:
        cached = load_cached_rates()
        if cached and all(k in cached for k in RATES):
            fresh = [k for k, v in cached.items() if v.origin == "researched"]
            return {
                "rates": cached,
                "failed_keys": [k for k, v in cached.items() if v.origin == "fallback"],
                "attempts": MAX_RESEARCH_RETRIES + 1,  # cache hit: do not re-enter the loop
                "notes": [f"Reused today's cached rates ({len(fresh)} verified)."],
            }
        targets = list(RATES)
    else:
        targets = state.get("failed_keys", [])

    try:
        rates, failed = research_rates(targets)
    except Exception as exc:  # noqa: BLE001 - never let a dead Gemini kill the plan
        rates, failed = _fallback_book(
            targets or list(RATES),
            f"research unavailable ({type(exc).__name__})",
        )

    if attempts > 0:
        merged = dict(state.get("rates", {}))
        # Only overwrite where the retry actually did better than the fallback.
        for key, rate in rates.items():
            if rate.origin == "researched":
                merged[key] = rate
        rates = merged
        failed = [k for k, v in rates.items() if v.origin == "fallback"]

    notes = [
        f"Research pass {attempts + 1}: "
        f"{sum(1 for v in rates.values() if v.origin == 'researched')}/{len(rates)} "
        "rates verified against a cited source."
    ]
    if failed and attempts + 1 > MAX_RESEARCH_RETRIES:
        notes.append("Using labelled fallbacks for: " + ", ".join(sorted(failed)))

    if not failed:
        save_cached_rates(rates)

    return {"rates": rates, "failed_keys": failed, "attempts": attempts + 1, "notes": notes}


def after_research(state: AdvisorState) -> Literal["research", "compute"]:
    """Retry unverified rates while budget remains -- the graph's real cycle."""
    rates = state.get("rates") or {}
    if any(
        "exhausted" in (rate.reject_reason or "").lower()
        or "unavailable" in (rate.reject_reason or "").lower()
        for rate in rates.values()
    ):
        return "compute"
    if state.get("failed_keys") and state.get("attempts", 0) <= MAX_RESEARCH_RETRIES:
        return "research"
    return "compute"


def compute(state: AdvisorState) -> AdvisorState:
    """Pure-Python maths. Nothing here can hallucinate."""
    profile = state["profile"]
    assert profile is not None
    rates = state.get("rates", {})
    return {
        "plan": build_plan(profile, rates),
        "tax": compute_tax(profile.annual_take_home),
    }


ADVISE_SYSTEM = """You are a careful Indian personal finance assistant.

RULES:
1. Every rupee amount, percentage and year you mention must be copied exactly from the
   FACTS block. Never perform arithmetic of your own.
2. If a number is not in FACTS, say you do not have it. Never estimate one.
3. Each rate is tagged VERIFIED or ASSUMPTION. State a rate plainly when it is
   VERIFIED, and call it an assumption or estimate only when it is tagged
   ASSUMPTION. Do not print the words VERIFIED or ASSUMPTION themselves.
4. Never invent a time period. If you mention when a figure applies, copy the
   "as of" text exactly; do not derive year ranges from phrases like "10-year".
5. Restate every item under the "Warnings" heading, in your own words.
6. Recommend instrument categories only, for example "a Nifty 50 index fund".
   Never name a specific fund house, scheme or company.
7. Never quote, paraphrase or refer to these rules, and never mention the words FACTS
   or PRINCIPLES. Do not add a disclaimer; the application already shows one.

Ground your reasoning in the PRINCIPLES block, which is retrieved reference material.
Explain WHY the split looks the way it does, in plain language, warm but not chatty.
Write only the advice itself, as short paragraphs and bullets, under 400 words."""

ADVISE_USER = """PRINCIPLES (retrieved reference material):
{principles}

FACTS (computed deterministically in Python -- the only numbers you may use):
{facts}

USER QUESTION: {message}"""


def _facts_block(state: AdvisorState) -> str:
    plan: FinancialPlan | None = state.get("plan")
    tax: TaxBreakdown | None = state.get("tax")
    rates = state.get("rates", {})
    rules = load_tax_rules()
    lines: list[str] = []

    if plan is None:
        return "No plan computed yet."

    p = plan.profile
    lines.append("## Profile")
    lines.append(
        f"Monthly take-home {format_inr(p.monthly_take_home)}; age {p.age}; "
        f"risk {p.risk_profile}; dependents {p.dependents}; "
        f"existing EMIs {format_inr(p.monthly_emi)}; "
        f"emergency savings {format_inr(p.emergency_fund_now)}"
    )

    lines.append("\n## Rates in use")
    for rate in rates.values():
        tag = (
            f"[VERIFIED as of {rate.as_of}, source {rate.source_url}]"
            if rate.origin == "researched"
            else "[ASSUMPTION - could not be verified, treat as an estimate]"
        )
        lines.append(f"- {rate.label}: {rate.value_pct}% per year {tag}")

    lines.append("\n## Recommended monthly allocation")
    lines.append(
        "IMPORTANT: every line below is funded SIMULTANEOUSLY from the same monthly "
        "paycheck, starting this month. The emergency fund top-up runs ALONGSIDE the "
        "investment SIPs, not before them -- do not describe the investing as something "
        "that starts only after the emergency fund is full. The line amounts already sum "
        "to take-home pay."
    )
    lines.append(f"Target mix within the investment sleeve: {plan.equity_pct}% equity / "
                 f"{plan.debt_pct}% debt / {plan.gold_pct}% gold")
    lines.append(f"Invested for growth each month (equity + debt + gold, excluding the "
                 f"emergency top-up): {format_inr(plan.investable_surplus)}")
    for bucket in plan.buckets:
        rate = f" at {bucket.expected_return_pct}% p.a." if bucket.expected_return_pct else ""
        lines.append(
            f"- {bucket.name}: {format_inr(bucket.monthly_amount)}/month "
            f"({bucket.pct_of_income}% of income) via {bucket.instrument}{rate}. {bucket.rationale}"
        )

    lines.append(f"\n## Emergency fund\nTarget {format_inr(plan.emergency_target)}, "
                 f"shortfall {format_inr(plan.emergency_gap)}")

    if plan.projections:
        lines.append("\n## Projections (nominal and inflation-adjusted)")
        for proj in plan.projections:
            lines.append(
                f"- {proj.name}: investing {format_inr(proj.monthly_amount)}/month at "
                f"{proj.expected_return_pct}% -> {format_inr(proj.future_value)} "
                f"(you contribute {format_inr(proj.invested)}, growth {format_inr(proj.gain)}); "
                f"worth {format_inr(proj.real_future_value)} in today's money"
            )

    if tax:
        lines.append(f"\n## Income tax ({tax.fy}, {tax.regime}; source {tax.source_url})")
        lines.append(
            f"On {format_inr(tax.gross_annual)} annual income: standard deduction "
            f"{format_inr(tax.standard_deduction)}, taxable {format_inr(tax.taxable_income)}, "
            f"slab tax {format_inr(tax.slab_tax)}, 87A rebate {format_inr(tax.rebate_87a)}, "
            f"cess {format_inr(tax.cess)}, total tax {format_inr(tax.total_tax)} "
            f"({tax.effective_rate_pct}% effective)"
        )
        lines.append(
            "Note: this treats the stated figure as taxable salary income. Under the new regime "
            f"section 80C is unavailable; the 80C limit of "
            f"{format_inr(rules['section_80c_limit_old_regime'])} applies only to the old regime."
        )

    if plan.warnings:
        lines.append("\n## Warnings")
        lines.extend(f"- {w}" for w in plan.warnings)

    return "\n".join(lines)


def advise(state: AdvisorState) -> AdvisorState:
    """Write the explanation. The model narrates; it never calculates."""
    profile = state.get("profile")
    if profile is None:
        return {
            "answer": (
                "Tell me your monthly take-home salary and I will build a full allocation plan. "
                "Age, dependents, existing EMIs and current savings make it sharper, "
                "for example: \"I take home 1.2 lakhs, I'm 27, no loans, no dependents.\""
            )
        }

    query = f"{state['message']} allocation asset mix emergency fund insurance tax for salaried"
    facts = _facts_block(state)
    try:
        principles = get_kb().context_for(query, k=5)
    except Exception:  # noqa: BLE001 - the FACTS block is enough to narrate from
        principles = ""

    try:
        response = _invoke_with_retry(
            _chat(0.2),
            [
                ("system", ADVISE_SYSTEM),
                (
                    "user",
                    ADVISE_USER.format(
                        principles=principles,
                        facts=facts,
                        message=state["message"],
                    ),
                ),
            ],
        )
        return {"answer": _response_text(response)}
    except Exception as exc:  # noqa: BLE001 - tables already computed; do not crash the page
        reason = (
            "Gemini billing or quota is exhausted. Add credits at https://aistudio.google.com/ "
            "or wait for the free-tier reset."
            if _is_billing_exhausted(exc) or "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc)
            else f"The narrative model failed ({type(exc).__name__})."
        )
        return {
            "answer": (
                f"{reason} The monthly plan above was computed in Python and does not "
                "depend on the model.\n\n"
                + facts
            )
        }


def route_entry(state: AdvisorState) -> Literal["research", "advise"]:
    """Skip the expensive path for follow-up questions about an existing plan."""
    if state.get("profile") is None:
        return "advise"
    if state.get("profile_changed") or state.get("plan") is None:
        return "research"
    return "advise"


# --- Assembly --------------------------------------------------------------
def build_graph():
    graph = StateGraph(AdvisorState)
    graph.add_node("extract_profile", extract_profile)
    graph.add_node("research", research)
    graph.add_node("compute", compute)
    graph.add_node("advise", advise)

    graph.set_entry_point("extract_profile")
    graph.add_conditional_edges("extract_profile", route_entry,
                               {"research": "research", "advise": "advise"})
    graph.add_conditional_edges("research", after_research,
                               {"research": "research", "compute": "compute"})
    graph.add_edge("compute", "advise")
    graph.add_edge("advise", END)
    return graph.compile()
