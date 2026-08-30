"""Deterministic financial engine. Pure Python, pure functions, zero LLM calls.

This module is the reason the project can claim accuracy. Language models are
used only to *retrieve* rates and to *explain* the output; every rupee figure a
user sees is computed here, so it is reproducible, unit-testable and auditable.
"""

from __future__ import annotations

import json
from functools import lru_cache

from config import DATA_DIR
from schemas import (
    Bucket,
    FinancialPlan,
    Projection,
    TaxBreakdown,
    UserProfile,
    ValidatedRate,
)

# --- Documented heuristics -------------------------------------------------
# These are rules of thumb, not statutory facts. They are named constants so an
# interviewer (or the user) can see and challenge every assumption.
ESSENTIALS_BASE_SHARE = 0.45          # share of take-home spent on needs, no dependents
ESSENTIALS_PER_DEPENDENT = 0.05       # added per dependent
ESSENTIALS_MAX_SHARE = 0.65
EMERGENCY_MONTHS = 6                  # months of essential outflow to hold liquid
EMERGENCY_MAX_SURPLUS_SHARE = 0.5     # never freeze more than half the surplus building it
TERM_COVER_MULTIPLE = 15              # life cover as a multiple of annual income
PPF_ANNUAL_CAP = 150_000              # statutory
RETIREMENT_AGE = 60

# Indicative annual term premium per rupee of cover, by age band.
TERM_PREMIUM_RATE = {29: 0.0009, 39: 0.0013, 49: 0.0022, 200: 0.0038}
# Indicative annual health premium for a 10 lakh floater.
HEALTH_PREMIUM_BASE = 12_000
HEALTH_PREMIUM_PER_DEPENDENT = 6_000

EQUITY_CAP = {"conservative": 40, "moderate": 70, "aggressive": 90}
EQUITY_FLOOR = {"conservative": 15, "moderate": 30, "aggressive": 45}
GOLD_SHARE = {"conservative": 10, "moderate": 10, "aggressive": 5}
# Split *within* the equity sleeve: (index, flexi-cap, mid & small cap)
EQUITY_SPLIT = {
    "conservative": (0.70, 0.30, 0.0),
    "moderate": (0.50, 0.30, 0.20),
    "aggressive": (0.40, 0.30, 0.30),
}


def format_inr(amount: float) -> str:
    """Format a rupee amount in the Indian lakh/crore digit grouping, e.g. 1,20,000."""
    negative = amount < 0
    whole = f"{abs(amount):,.0f}".replace(",", "")
    if len(whole) > 3:
        head, tail = whole[:-3], whole[-3:]
        groups = []
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        if head:
            groups.insert(0, head)
        whole = ",".join(groups) + "," + tail
    return ("-" if negative else "") + "Rs " + whole


# --- Tax -------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_tax_rules() -> dict:
    return json.loads((DATA_DIR / "tax_rules.json").read_text(encoding="utf-8"))


def compute_tax(gross_annual: float, rules: dict | None = None) -> TaxBreakdown:
    """Income tax under India's new regime for a salaried individual.

    Implements progressive slabs, the standard deduction, the section 87A rebate
    *including* its marginal relief, surcharge and health & education cess.
    Surcharge marginal relief is not modelled -- it only bites within a narrow
    band just above 50 lakh of taxable income.
    """
    rules = rules or load_tax_rules()
    std_deduction = float(rules["standard_deduction"])
    taxable = max(0.0, gross_annual - std_deduction)

    slab_tax = 0.0
    lower = 0.0
    for slab in rules["slabs"]:
        upper = slab["upto"]
        ceiling = taxable if upper is None else min(taxable, float(upper))
        if ceiling > lower:
            slab_tax += (ceiling - lower) * float(slab["rate"])
            lower = ceiling
        if upper is not None and taxable <= float(upper):
            break

    # Section 87A. Below the limit the rebate wipes out the tax; just above it,
    # marginal relief caps total tax at the income earned beyond the limit.
    rebate_cfg = rules["rebate_87a"]
    limit = float(rebate_cfg["taxable_income_limit"])
    max_rebate = float(rebate_cfg["max_rebate"])
    if taxable <= limit:
        rebate = min(slab_tax, max_rebate)
    else:
        excess = taxable - limit
        rebate = min(max(0.0, slab_tax - excess), max_rebate)

    tax_after_rebate = slab_tax - rebate

    surcharge_rate = 0.0
    for band in rules["surcharge_slabs"]:
        if taxable > float(band["above"]):
            surcharge_rate = float(band["rate"])
            break
    surcharge = tax_after_rebate * surcharge_rate

    cess = (tax_after_rebate + surcharge) * float(rules["cess_rate"])
    total = tax_after_rebate + surcharge + cess

    return TaxBreakdown(
        fy=rules["fy"],
        source_url=rules["source_url"],
        regime=rules["regime"],
        gross_annual=round(gross_annual, 2),
        standard_deduction=std_deduction,
        taxable_income=round(taxable, 2),
        slab_tax=round(slab_tax, 2),
        rebate_87a=round(rebate, 2),
        surcharge=round(surcharge, 2),
        cess=round(cess, 2),
        total_tax=round(total, 2),
        effective_rate_pct=round(total / gross_annual * 100, 2) if gross_annual else 0.0,
    )


# --- Compounding -----------------------------------------------------------
def sip_future_value(monthly: float, annual_rate_pct: float, years: int) -> float:
    """Future value of a monthly SIP, contributions at the start of each month."""
    n = years * 12
    i = annual_rate_pct / 100.0 / 12.0
    if n <= 0:
        return 0.0
    if abs(i) < 1e-12:
        return monthly * n
    return monthly * (((1 + i) ** n - 1) / i) * (1 + i)


def real_value(nominal: float, inflation_pct: float, years: int) -> float:
    """Discount a future amount back into today's purchasing power."""
    return nominal / ((1 + inflation_pct / 100.0) ** years)


# --- Allocation ------------------------------------------------------------
def estimate_essentials(profile: UserProfile) -> tuple[float, bool]:
    if profile.monthly_essentials is not None:
        return profile.monthly_essentials, False
    share = min(
        ESSENTIALS_BASE_SHARE + ESSENTIALS_PER_DEPENDENT * profile.dependents,
        ESSENTIALS_MAX_SHARE,
    )
    return round(profile.monthly_take_home * share, 2), True


def asset_mix(age: int, risk: str) -> tuple[int, int, int]:
    """Equity / debt / gold percentages from an age glidepath, clamped by risk band."""
    equity = max(EQUITY_FLOOR[risk], min(EQUITY_CAP[risk], 100 - age))
    gold = GOLD_SHARE[risk]
    debt = 100 - equity - gold
    return equity, debt, gold


def _term_premium_monthly(profile: UserProfile) -> float:
    cover = profile.annual_take_home * TERM_COVER_MULTIPLE
    rate = next(r for age_cap, r in sorted(TERM_PREMIUM_RATE.items()) if profile.age <= age_cap)
    return round(cover * rate / 12, 2)


def _health_premium_monthly(profile: UserProfile) -> float:
    annual = HEALTH_PREMIUM_BASE + HEALTH_PREMIUM_PER_DEPENDENT * profile.dependents
    if profile.age >= 40:
        annual *= 1.6
    return round(annual / 12, 2)


def build_plan(profile: UserProfile, rates: dict[str, ValidatedRate]) -> FinancialPlan:
    """Turn a profile plus validated rates into a fully specified monthly plan.

    Surplus is allocated as a priority waterfall -- protection and liquidity are
    funded before growth, because an equity SIP that gets redeemed in an
    emergency is worse than no SIP at all.
    """
    income = profile.monthly_take_home
    warnings: list[str] = []

    essentials, estimated = estimate_essentials(profile)
    if estimated:
        warnings.append(
            f"Essential spend was not provided, so it is estimated at "
            f"{format_inr(essentials)}/month from income and dependents. "
            "Tell me your actual figure for a sharper plan."
        )

    insurance = 0.0
    if not profile.has_term_insurance:
        premium = _term_premium_monthly(profile)
        insurance += premium
        cover = profile.annual_take_home * TERM_COVER_MULTIPLE
        warnings.append(
            f"No term life cover. Provisioned {format_inr(premium)}/month for roughly "
            f"Rs {cover / 1e7:.1f} crore of cover ({TERM_COVER_MULTIPLE}x annual income). "
            "This is an indicative estimate -- get real quotes."
        )
    if not profile.has_health_insurance:
        premium = _health_premium_monthly(profile)
        insurance += premium
        warnings.append(
            f"No health cover. Provisioned {format_inr(premium)}/month for an indicative "
            "Rs 10 lakh floater."
        )

    if profile.monthly_emi > 0.4 * income:
        warnings.append(
            f"EMIs are {profile.monthly_emi / income * 100:.0f}% of take-home, above the 40% "
            "comfort ceiling. Prepaying debt likely beats new investing."
        )

    available = income - essentials - profile.monthly_emi - insurance

    emergency_target = round(EMERGENCY_MONTHS * (essentials + profile.monthly_emi), 2)
    emergency_gap = max(0.0, emergency_target - profile.emergency_fund_now)

    buckets: list[Bucket] = []

    def add(name: str, instrument: str, amount: float, rate_key: str, rationale: str,
            horizon: int | None = None) -> None:
        if amount < 100:  # not worth a line item
            return
        rate = rates.get(rate_key)
        buckets.append(
            Bucket(
                name=name,
                instrument=instrument,
                monthly_amount=round(amount, 2),
                pct_of_income=round(amount / income * 100, 1),
                rate_key=rate_key,
                expected_return_pct=rate.value_pct if rate else None,
                rationale=rationale,
                horizon_years=horizon,
            )
        )

    add("Essentials", "Rent, food, utilities, transport", essentials, "",
        "Non-negotiable cash outflow, kept under the 50% guideline where possible.")
    if profile.monthly_emi:
        add("Existing EMIs", "Loan repayments", profile.monthly_emi, "",
            "Contractual obligation, serviced before any investing.")
    if insurance:
        add("Insurance premiums", "Term life + health floater", insurance, "",
            "Protection first: one hospital bill can undo years of SIPs.")

    if available <= 0:
        warnings.append(
            "Essentials, EMIs and insurance already exceed take-home pay. There is no "
            "investable surplus -- the priority is cutting fixed costs or raising income, "
            "not choosing funds."
        )
        return FinancialPlan(
            profile=profile, assumed_essentials=essentials, essentials_estimated=estimated,
            emergency_target=emergency_target, emergency_gap=round(emergency_gap, 2),
            investable_surplus=0.0, buckets=buckets, projections=[],
            equity_pct=0, debt_pct=0, gold_pct=0, warnings=warnings,
        )

    emergency_topup = min(emergency_gap, EMERGENCY_MAX_SURPLUS_SHARE * available)
    if emergency_topup > 0:
        months = emergency_gap / emergency_topup
        add("Emergency fund top-up", "Sweep-in FD / liquid fund", emergency_topup, "bank_savings",
            f"Building {EMERGENCY_MONTHS} months of expenses ({format_inr(emergency_target)}). "
            f"Fully funded in about {months:.0f} months at this rate.", horizon=1)

    investable = available - emergency_topup
    equity_pct, debt_pct, gold_pct = asset_mix(profile.age, profile.risk_profile)
    retirement_years = max(1, RETIREMENT_AGE - profile.age)

    equity_amt = investable * equity_pct / 100
    debt_amt = investable * debt_pct / 100
    gold_amt = investable * gold_pct / 100

    index_w, flexi_w, small_w = EQUITY_SPLIT[profile.risk_profile]
    add("Equity - core", "Nifty 50 / broad-market index fund SIP", equity_amt * index_w,
        "nifty50_10y_cagr",
        "Lowest-cost way to own the market. Core of the growth sleeve.", horizon=retirement_years)
    add("Equity - flexi cap", "Flexi-cap active fund SIP", equity_amt * flexi_w,
        "nifty50_10y_cagr",
        "Manager discretion across market caps, using the index CAGR as a neutral "
        "return assumption rather than an invented alpha figure.", horizon=retirement_years)
    add("Equity - mid & small cap", "Mid/small-cap index or fund SIP", equity_amt * small_w,
        "nifty50_10y_cagr",
        "Higher volatility for a long horizon. Deliberately assumes the same expected "
        "return as the index -- extra risk is not a guarantee of extra return.",
        horizon=retirement_years)

    ppf_amt = min(debt_amt * 0.5, PPF_ANNUAL_CAP / 12)
    add("Debt - PPF", "Public Provident Fund", ppf_amt, "ppf",
        f"Sovereign-backed and tax-free on maturity. Capped at the statutory "
        f"{format_inr(PPF_ANNUAL_CAP)}/year, i.e. {format_inr(PPF_ANNUAL_CAP / 12)}/month. "
        "15-year lock-in.", horizon=15)
    add("Debt - flexible", "Short-duration debt fund or FD ladder", debt_amt - ppf_amt, "debt_fund",
        "Keeps part of the debt sleeve accessible, unlike PPF's lock-in.", horizon=5)
    add("Gold", "Gold ETF or gold index fund", gold_amt, "gold_10y_cagr",
        "Portfolio hedge, not a growth engine. Sovereign Gold Bonds are no longer "
        "issued, so an ETF is the practical route.", horizon=10)

    invest_buckets = [b for b in buckets if b.horizon_years and b.expected_return_pct is not None
                      and b.name not in ("Essentials", "Existing EMIs", "Insurance premiums")]
    growth = [b for b in invest_buckets if b.name != "Emergency fund top-up"]
    total_growth = sum(b.monthly_amount for b in growth)
    blended = (
        sum(b.monthly_amount * (b.expected_return_pct or 0) for b in growth) / total_growth
        if total_growth else 0.0
    )

    inflation = rates.get("inflation_cpi")
    inflation_pct = inflation.value_pct if inflation else 5.0

    projections: list[Projection] = []
    for years in sorted({5, 10, 20, retirement_years}):
        fv = sip_future_value(total_growth, blended, years)
        invested = total_growth * years * 12
        projections.append(
            Projection(
                name=f"Portfolio after {years} year{'s' if years > 1 else ''}"
                + (f" (age {profile.age + years})" if years == retirement_years else ""),
                monthly_amount=round(total_growth, 2),
                years=years,
                expected_return_pct=round(blended, 2),
                invested=round(invested, 2),
                future_value=round(fv, 2),
                gain=round(fv - invested, 2),
                real_future_value=round(real_value(fv, inflation_pct, years), 2),
            )
        )

    fallbacks = sorted(k for k, v in rates.items() if v.origin == "fallback")
    if fallbacks:
        warnings.append(
            "Live lookup failed validation for: " + ", ".join(fallbacks)
            + ". Labelled fallback assumptions were used for those, not researched figures."
        )

    return FinancialPlan(
        profile=profile,
        assumed_essentials=essentials,
        essentials_estimated=estimated,
        emergency_target=emergency_target,
        emergency_gap=round(emergency_gap, 2),
        investable_surplus=round(investable, 2),
        buckets=buckets,
        projections=projections,
        equity_pct=equity_pct,
        debt_pct=debt_pct,
        gold_pct=gold_pct,
        warnings=warnings,
    )
