"""Tests for the anti-hallucination guardrail.

`validate_rates` is the single choke point every researched number must pass.
These tests encode the project's core claim: a number the model could not cite
never reaches the maths engine.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RATES  # noqa: E402
from finance import build_plan  # noqa: E402
from research import _number_supported_by_evidence, validate_rates  # noqa: E402
from schemas import RateResearchResult, ResearchedRate, UserProfile, ValidatedRate  # noqa: E402


def _full_book() -> dict[str, ValidatedRate]:
    return {
        key: ValidatedRate(key=key, label=spec.label, value_pct=spec.fallback,
                           origin="researched")
        for key, spec in RATES.items()
    }


def make_rate(**kwargs) -> ResearchedRate:
    defaults = dict(
        key="ppf",
        value_pct=7.1,
        label="PPF",
        as_of="Q2 FY26",
        source_url="https://example.gov.in/ppf",
        evidence="The PPF interest rate remains unchanged at 7.1% for the quarter.",
    )
    return ResearchedRate(**{**defaults, **kwargs})


# --- Evidence matching -----------------------------------------------------
def test_number_found_in_evidence() -> None:
    assert _number_supported_by_evidence(7.1, "PPF stays at 7.1% this quarter")
    assert _number_supported_by_evidence(8.25, "EPFO declared 8.25 per cent for the year")
    assert _number_supported_by_evidence(12.0, "The index returned 12 percent annually")


def test_number_absent_from_evidence_is_rejected() -> None:
    assert not _number_supported_by_evidence(9.5, "PPF stays at 7.1% this quarter")
    assert not _number_supported_by_evidence(7.1, "")
    assert not _number_supported_by_evidence(7.1, "no digits at all here")


def test_thousands_separators_do_not_break_matching() -> None:
    assert _number_supported_by_evidence(7.1, "Rate of 7.1% on deposits above 1,00,000")


# --- Validation ------------------------------------------------------------
def test_well_cited_rate_is_accepted() -> None:
    rates, failed = validate_rates(RateResearchResult(rates=[make_rate()]), ["ppf"])
    assert failed == []
    assert rates["ppf"].origin == "researched"
    assert rates["ppf"].value_pct == 7.1


def test_out_of_band_value_falls_back() -> None:
    """A "45% PPF rate" is the classic hallucination this bound exists to catch."""
    absurd = make_rate(value_pct=45.0, evidence="PPF now pays a stunning 45% return")
    rates, failed = validate_rates(RateResearchResult(rates=[absurd]), ["ppf"])
    assert failed == ["ppf"]
    assert rates["ppf"].origin == "fallback"
    assert "outside the plausible" in rates["ppf"].reject_reason


def test_uncited_number_falls_back_even_when_plausible() -> None:
    """7.4% is a perfectly believable PPF rate -- and still rejected without evidence."""
    uncited = make_rate(value_pct=7.4, evidence="The PPF rate was left unchanged at 7.1%.")
    rates, failed = validate_rates(RateResearchResult(rates=[uncited]), ["ppf"])
    assert failed == ["ppf"]
    assert "does not appear in the quoted evidence" in rates["ppf"].reject_reason


def test_missing_source_url_falls_back() -> None:
    rates, failed = validate_rates(
        RateResearchResult(rates=[make_rate(source_url="the RBI website")]), ["ppf"]
    )
    assert failed == ["ppf"]
    assert rates["ppf"].reject_reason == "no usable source URL"


def test_unknown_keys_are_ignored_and_omissions_fall_back() -> None:
    result = RateResearchResult(rates=[make_rate(key="crypto_moon_rate", value_pct=900)])
    rates, failed = validate_rates(result, ["ppf", "debt_fund"])
    assert "crypto_moon_rate" not in rates
    assert sorted(failed) == ["debt_fund", "ppf"]
    assert all(r.origin == "fallback" for r in rates.values())
    assert rates["ppf"].reject_reason == "not returned by research"


def test_rate_book_is_always_complete() -> None:
    """Downstream maths must never see a missing key, whatever the crew returns."""
    keys = list(RATES)
    rates, _ = validate_rates(RateResearchResult(), keys)
    assert set(rates) == set(keys)
    assert all(r.value_pct > 0 for r in rates.values())


def test_every_rate_in_the_table_is_actually_used_by_the_engine() -> None:
    """Each rate costs a web search, so an unused one is pure waste."""
    used = {b.rate_key for b in build_plan(
        UserProfile(monthly_take_home=120_000, monthly_emi=5_000), _full_book()
    ).buckets if b.rate_key} | {"inflation_cpi"}  # inflation feeds the real-value maths
    assert set(RATES) == used


def test_first_valid_result_wins_over_duplicates() -> None:
    good = make_rate(value_pct=7.1)
    later = make_rate(value_pct=8.9, evidence="PPF is 8.9% now")
    rates, _ = validate_rates(RateResearchResult(rates=[good, later]), ["ppf"])
    assert rates["ppf"].value_pct == 7.1
