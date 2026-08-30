"""Offline tests for graph wiring and prompt assembly.

The LLM nodes are stubbed, so this verifies the parts that must hold regardless
of what the model says: routing decisions, the retry cycle, and — most
importantly — that the FACTS block handed to the advisor actually contains the
numbers it is told to copy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import graph as G  # noqa: E402
from config import RATES  # noqa: E402
from schemas import UserProfile, ValidatedRate  # noqa: E402


@pytest.fixture
def rates() -> dict[str, ValidatedRate]:
    return {
        key: ValidatedRate(
            key=key, label=key, value_pct=spec.fallback, origin="researched",
            as_of="Aug 2026", source_url=f"https://example.com/{key}",
            evidence=f"The rate is {spec.fallback}%.",
        )
        for key, spec in RATES.items()
    }


@pytest.fixture
def computed(rates) -> dict:
    profile = UserProfile(monthly_take_home=120_000, age=27, monthly_essentials=40_000)
    state = {"profile": profile, "rates": rates, "message": "how should I invest?"}
    state.update(G.compute(state))
    return state


# --- Routing ---------------------------------------------------------------
def test_no_profile_routes_straight_to_advice() -> None:
    assert G.route_entry({"profile": None}) == "advise"


def test_new_facts_trigger_research() -> None:
    profile = UserProfile(monthly_take_home=100_000)
    assert G.route_entry({"profile": profile, "profile_changed": True, "plan": None}) == "research"


def test_followup_with_unchanged_profile_skips_research(computed) -> None:
    """The cheap path: no new facts and a plan already in hand."""
    state = {**computed, "profile_changed": False}
    assert G.route_entry(state) == "advise"


def test_retry_cycle_engages_only_while_budget_remains() -> None:
    assert G.after_research({"failed_keys": ["ppf"], "attempts": 1}) == "research"
    assert G.after_research({"failed_keys": ["ppf"], "attempts": 99}) == "compute"
    assert G.after_research({"failed_keys": [], "attempts": 1}) == "compute"


# --- Compute node ----------------------------------------------------------
def test_compute_produces_plan_and_tax(computed) -> None:
    plan, tax = computed["plan"], computed["tax"]
    assert plan.investable_surplus > 0
    assert plan.equity_pct + plan.debt_pct + plan.gold_pct == 100
    assert tax.gross_annual == pytest.approx(1_440_000)
    assert tax.total_tax == pytest.approx(88_140, abs=1)


# --- FACTS block -----------------------------------------------------------
def test_facts_block_contains_every_number_the_advisor_may_use(computed) -> None:
    facts = G._facts_block(computed)
    for section in ("## Profile", "## Rates in use",
                    "## Recommended monthly allocation", "## Projections", "## Income tax"):
        assert section in facts

    plan = computed["plan"]
    from finance import format_inr

    assert format_inr(plan.investable_surplus) in facts
    for bucket in plan.buckets:
        assert format_inr(bucket.monthly_amount) in facts
    for projection in plan.projections:
        assert format_inr(projection.future_value) in facts
        assert format_inr(projection.real_future_value) in facts


def test_facts_block_labels_fallbacks_and_cites_sources(computed, rates) -> None:
    """Verified and unverified rates must be distinguishable, or the model mislabels them."""
    clean = G._facts_block(computed)
    assert "[ASSUMPTION" not in clean
    assert clean.count("[VERIFIED as of") == len(rates)

    degraded = dict(rates)
    degraded["ppf"] = ValidatedRate(key="ppf", label="PPF", value_pct=7.1, origin="fallback")
    state = {**computed, "rates": degraded}
    state.update(G.compute(state))
    facts = G._facts_block(state)
    assert "[ASSUMPTION" in facts
    assert "https://example.com/debt_fund" in facts


def test_facts_block_propagates_warnings(computed) -> None:
    profile = UserProfile(monthly_take_home=30_000, monthly_essentials=25_000, monthly_emi=10_000)
    state = {**computed, "profile": profile}
    state.update(G.compute(state))
    facts = G._facts_block(state)
    assert "## Warnings" in facts
    assert "exceed take-home" in facts


def test_facts_block_is_safe_before_any_plan_exists() -> None:
    assert G._facts_block({"message": "hi"}) == "No plan computed yet."


# --- Advise node fallback --------------------------------------------------
def test_advise_asks_for_salary_when_profile_is_missing() -> None:
    """Must not call the LLM or the knowledge base when there is nothing to advise on."""
    answer = G.advise({"profile": None, "message": "hello"})["answer"]
    assert "take-home salary" in answer


# --- Response flattening ---------------------------------------------------
class _Msg:
    def __init__(self, content, text=None):
        self.content = content
        if text is not None:
            self.text = text


def test_plain_string_content() -> None:
    assert G._response_text(_Msg("hello")) == "hello"


def test_gemini_block_list_content_is_flattened() -> None:
    """Gemini returns typed blocks; rendering their repr would leak signatures to the user."""
    blocks = [
        {"type": "text", "text": "first", "extras": {"signature": "abc"}},
        {"type": "text", "text": "second"},
    ]
    assert G._response_text(_Msg(blocks)) == "first\nsecond"


def test_text_property_and_method_both_supported() -> None:
    assert G._response_text(_Msg([], text="from property")) == "from property"
    assert G._response_text(_Msg([], text=lambda: "from method")) == "from method"


def test_callable_string_shim_is_read_as_a_string() -> None:
    """langchain-core's back-compat `.text` is both a str and callable.

    It must be read as a string, since calling it emits a deprecation warning.
    """

    class _CallableStr(str):
        def __call__(self):
            raise AssertionError("should not be called")

    assert G._response_text(_Msg([], text=_CallableStr("shim value"))) == "shim value"


def test_empty_response_is_empty_string_not_a_crash() -> None:
    assert G._response_text(_Msg([])) == ""
    assert G._response_text(_Msg(None)) == ""


# --- Profile extraction guards --------------------------------------------
def test_lakh_shorthand_is_expanded(monkeypatch) -> None:
    """A model returning 1.2 for "1.2 lakhs" is corrected in Python, not trusted."""
    from schemas import ProfileExtraction

    monkeypatch.setattr(G, "_chat", lambda *_, **__: _StubLLM(ProfileExtraction(salary_amount=1.2)))
    result = G.extract_profile({"message": "my salary is 1.2 lakhs"})
    assert result["profile"].monthly_take_home == pytest.approx(120_000)
    assert result["profile_changed"] is True


def test_annual_salary_is_converted_to_monthly(monkeypatch) -> None:
    from schemas import ProfileExtraction

    extraction = ProfileExtraction(salary_amount=2_400_000, salary_is_annual=True, age=35)
    monkeypatch.setattr(G, "_chat", lambda *_, **__: _StubLLM(extraction))
    profile = G.extract_profile({"message": "I earn 24 LPA, age 35"})["profile"]
    assert profile.monthly_take_home == pytest.approx(200_000)
    assert profile.age == 35


def test_stated_facts_merge_onto_existing_profile(monkeypatch) -> None:
    from schemas import ProfileExtraction

    monkeypatch.setattr(
        G, "_chat", lambda *_, **__: _StubLLM(ProfileExtraction(monthly_emi=15_000))
    )
    previous = UserProfile(monthly_take_home=120_000, age=27)
    result = G.extract_profile({"message": "I also have a 15k EMI", "profile": previous})
    assert result["profile"].monthly_take_home == pytest.approx(120_000)  # preserved
    assert result["profile"].monthly_emi == 15_000                        # added
    assert result["profile_changed"] is True


def test_message_with_no_facts_leaves_profile_untouched(monkeypatch) -> None:
    from schemas import ProfileExtraction

    monkeypatch.setattr(G, "_chat", lambda *_, **__: _StubLLM(ProfileExtraction()))
    previous = UserProfile(monthly_take_home=120_000, age=27)
    result = G.extract_profile({"message": "why gold?", "profile": previous})
    assert result["profile_changed"] is False


def test_extraction_failure_says_so_instead_of_blaming_the_user(monkeypatch) -> None:
    """An API outage must not be reported as "you didn't give me a salary"."""
    monkeypatch.setattr(G, "_chat", lambda *_, **__: _RaisingLLM())
    result = G.extract_profile({"message": "my salary is 1.2 lakhs", "profile": None})
    assert result["profile"] is None and result["profile_changed"] is False
    assert "Could not read new details" in result["notes"][0]


def test_extraction_failure_keeps_an_existing_profile(monkeypatch) -> None:
    monkeypatch.setattr(G, "_chat", lambda *_, **__: _RaisingLLM())
    previous = UserProfile(monthly_take_home=120_000, age=27)
    result = G.extract_profile({"message": "why gold?", "profile": previous})
    assert result["profile"] == previous
    assert "existing plan" in result["notes"][0]


# --- Rate-limit backoff ----------------------------------------------------
def test_rate_limit_detection() -> None:
    assert G._is_rate_limit(RuntimeError("429 RESOURCE_EXHAUSTED"))
    assert G._is_rate_limit(RuntimeError("You exceeded your current quota"))
    assert not G._is_rate_limit(RuntimeError("404 NOT_FOUND"))


def test_retry_delay_prefers_the_servers_own_hint() -> None:
    assert G._retry_delay(RuntimeError("Please retry in 53.5s."), 0) == pytest.approx(54.5)
    assert G._retry_delay(RuntimeError("boom"), 0) == 5.0     # exponential fallback
    assert G._retry_delay(RuntimeError("boom"), 3) == 40.0
    assert G._retry_delay(RuntimeError("Please retry in 999s."), 0) == 90.0  # capped


def test_retry_succeeds_after_a_transient_rate_limit(monkeypatch) -> None:
    monkeypatch.setattr(G.time, "sleep", lambda _s: None)
    calls = {"n": 0}

    class _Flaky:
        def invoke(self, _payload):
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("429 RESOURCE_EXHAUSTED. Please retry in 1s.")
            return "recovered"

    assert G._invoke_with_retry(_Flaky(), "x") == "recovered"
    assert calls["n"] == 3


def test_non_rate_limit_errors_are_not_retried() -> None:
    calls = {"n": 0}

    class _Broken:
        def invoke(self, _payload):
            calls["n"] += 1
            raise ValueError("bad request")

    with pytest.raises(ValueError):
        G._invoke_with_retry(_Broken(), "x")
    assert calls["n"] == 1  # retrying a bad request would only waste quota


class _StubLLM:
    def __init__(self, payload):
        self.payload = payload

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _prompt):
        return self.payload


class _RaisingLLM(_StubLLM):
    def __init__(self):
        super().__init__(None)

    def invoke(self, _prompt):
        raise RuntimeError("API down")
