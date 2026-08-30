"""Tests for the deterministic layer.

Only the parts that must be exactly right are tested here: tax arithmetic,
compounding, allocation invariants and the evidence guardrail. Nothing in this
file touches an API, so the whole suite runs offline in under a second (bar the
one-off crewai import).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config import RATES  # noqa: E402
from finance import (  # noqa: E402
    PPF_ANNUAL_CAP,
    asset_mix,
    build_plan,
    compute_tax,
    estimate_essentials,
    format_inr,
    real_value,
    sip_future_value,
)
from schemas import UserProfile, ValidatedRate  # noqa: E402

STD_DEDUCTION = 75_000


def rate_book(**overrides: float) -> dict[str, ValidatedRate]:
    values = {key: spec.fallback for key, spec in RATES.items()} | overrides
    return {
        key: ValidatedRate(key=key, label=key, value_pct=value, origin="researched")
        for key, value in values.items()
    }


# --- Tax -------------------------------------------------------------------
@pytest.mark.parametrize(
    ("taxable", "expected_total"),
    [
        (400_000, 0),          # entirely within the nil slab
        (800_000, 0),          # 20,000 slab tax fully wiped by the 87A rebate
        (1_200_000, 0),        # exactly at the rebate ceiling
        (1_210_000, 10_400),   # marginal relief caps tax at the 10,000 excess, plus 4% cess
        (1_365_000, 88_140),   # 1.2 lakh/month salary: rebate no longer applies
        (2_000_000, 208_000),
    ],
)
def test_new_regime_tax(taxable: float, expected_total: float) -> None:
    tax = compute_tax(taxable + STD_DEDUCTION)
    assert tax.taxable_income == pytest.approx(taxable)
    assert tax.total_tax == pytest.approx(expected_total, abs=1.0)


def test_rebate_boundary_is_continuous() -> None:
    """Marginal relief must stop the cliff at the 12 lakh rebate limit."""
    just_below = compute_tax(1_200_000 + STD_DEDUCTION).total_tax
    just_above = compute_tax(1_201_000 + STD_DEDUCTION).total_tax
    assert just_below == 0
    assert just_above < 2_000  # would be ~63,000 without marginal relief


def test_surcharge_applies_above_fifty_lakh() -> None:
    tax = compute_tax(6_000_000 + STD_DEDUCTION)
    assert tax.surcharge > 0
    assert tax.effective_rate_pct < 30  # progressive, so never the headline rate


def test_zero_income_is_safe() -> None:
    tax = compute_tax(0)
    assert tax.total_tax == 0
    assert tax.effective_rate_pct == 0


# --- Compounding -----------------------------------------------------------
def test_sip_future_value_known_case() -> None:
    """10k/month at 12% for 10 years is a widely quoted ~23.2 lakh."""
    assert sip_future_value(10_000, 12.0, 10) == pytest.approx(2_323_391, rel=1e-4)


def test_sip_with_zero_return_is_just_the_contributions() -> None:
    assert sip_future_value(5_000, 0.0, 3) == pytest.approx(180_000)


def test_sip_is_monotonic_in_rate_and_horizon() -> None:
    assert sip_future_value(1000, 8, 10) > sip_future_value(1000, 6, 10)
    assert sip_future_value(1000, 8, 20) > sip_future_value(1000, 8, 10)


def test_real_value_erodes_with_inflation() -> None:
    assert real_value(100.0, 6.0, 12) == pytest.approx(49.7, rel=1e-2)


# --- Allocation ------------------------------------------------------------
@pytest.mark.parametrize(
    ("age", "risk", "expected"),
    [
        (30, "moderate", (70, 20, 10)),
        (25, "conservative", (40, 50, 10)),
        (50, "aggressive", (50, 45, 5)),
        (70, "conservative", (30, 60, 10)),
    ],
)
def test_asset_mix(age: int, risk: str, expected: tuple[int, int, int]) -> None:
    assert asset_mix(age, risk) == expected


def test_asset_mix_always_sums_to_100_and_stays_positive() -> None:
    for age in range(18, 76):
        for risk in ("conservative", "moderate", "aggressive"):
            equity, debt, gold = asset_mix(age, risk)
            assert equity + debt + gold == 100
            assert min(equity, debt, gold) >= 0


def test_essentials_estimate_scales_with_dependents_and_is_capped() -> None:
    base, estimated = estimate_essentials(UserProfile(monthly_take_home=100_000))
    assert estimated and base == pytest.approx(45_000)
    with_kids, _ = estimate_essentials(UserProfile(monthly_take_home=100_000, dependents=2))
    assert with_kids > base
    capped, _ = estimate_essentials(UserProfile(monthly_take_home=100_000, dependents=10))
    assert capped == pytest.approx(65_000)
    stated, estimated = estimate_essentials(
        UserProfile(monthly_take_home=100_000, monthly_essentials=30_000)
    )
    assert stated == 30_000 and not estimated


# --- Plan invariants -------------------------------------------------------
def test_plan_allocates_the_whole_salary_and_nothing_more() -> None:
    profile = UserProfile(monthly_take_home=120_000, age=27, monthly_essentials=40_000)
    plan = build_plan(profile, rate_book())
    total = sum(b.monthly_amount for b in plan.buckets)
    assert total <= profile.monthly_take_home + 1
    assert total == pytest.approx(profile.monthly_take_home, abs=500)
    assert plan.investable_surplus > 0


def test_emergency_fund_is_funded_before_full_investing() -> None:
    """With no savings, part of the surplus must be diverted to liquidity."""
    profile = UserProfile(monthly_take_home=120_000, age=27, monthly_essentials=40_000)
    empty = build_plan(profile, rate_book())
    assert any(b.name == "Emergency fund top-up" for b in empty.buckets)

    funded = build_plan(
        profile.model_copy(update={"emergency_fund_now": 1_000_000}), rate_book()
    )
    assert not any(b.name == "Emergency fund top-up" for b in funded.buckets)
    assert funded.investable_surplus > empty.investable_surplus


def test_ppf_respects_the_statutory_cap() -> None:
    profile = UserProfile(
        monthly_take_home=1_000_000, age=55, risk_profile="conservative",
        monthly_essentials=100_000, emergency_fund_now=10_000_000,
        has_term_insurance=True, has_health_insurance=True,
    )
    plan = build_plan(profile, rate_book())
    ppf = next(b for b in plan.buckets if b.name == "Debt - PPF")
    assert ppf.monthly_amount <= PPF_ANNUAL_CAP / 12 + 0.01


def test_negative_surplus_refuses_to_invest() -> None:
    profile = UserProfile(
        monthly_take_home=30_000, monthly_essentials=25_000, monthly_emi=10_000
    )
    plan = build_plan(profile, rate_book())
    assert plan.investable_surplus == 0
    assert plan.projections == []
    assert any("exceed take-home" in w for w in plan.warnings)


def test_fallback_rates_are_surfaced_as_a_warning() -> None:
    rates = rate_book()
    rates["ppf"] = ValidatedRate(key="ppf", label="PPF", value_pct=7.1, origin="fallback")
    plan = build_plan(UserProfile(monthly_take_home=100_000), rates)
    assert any("fallback" in w for w in plan.warnings)


def test_missing_insurance_produces_a_provision_and_a_warning() -> None:
    profile = UserProfile(monthly_take_home=100_000, monthly_essentials=30_000)
    plan = build_plan(profile, rate_book())
    assert any(b.name == "Insurance premiums" for b in plan.buckets)
    assert any("term life cover" in w for w in plan.warnings)

    covered = build_plan(
        profile.model_copy(update={"has_term_insurance": True, "has_health_insurance": True}),
        rate_book(),
    )
    assert not any(b.name == "Insurance premiums" for b in covered.buckets)


# --- Formatting ------------------------------------------------------------
@pytest.mark.parametrize(
    ("amount", "expected"),
    [
        (999, "Rs 999"),
        (1_234, "Rs 1,234"),
        (120_000, "Rs 1,20,000"),
        (10_000_000, "Rs 1,00,00,000"),
        (-50_000, "-Rs 50,000"),
    ],
)
def test_indian_number_formatting(amount: float, expected: str) -> None:
    assert format_inr(amount) == expected
