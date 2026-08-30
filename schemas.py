"""Typed contracts shared by every layer.

Structured output is the first line of defence against hallucination: the model
is never allowed to return free-form numbers, only instances of these models.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

RiskProfile = Literal["conservative", "moderate", "aggressive"]


class ResearchedRate(BaseModel):
    """A single live rate together with the evidence that justifies it.

    `source_url` and `evidence` are mandatory. A rate with no verbatim snippet
    backing it is treated as fabricated and dropped during validation.
    """

    key: str
    value_pct: float = Field(description="Annual rate as a percentage, e.g. 7.1 for 7.1%")
    label: str = Field(description="Human readable name of the instrument")
    as_of: str = Field(description="Period the figure applies to, e.g. 'Q2 FY26' or 'Aug 2026'")
    source_url: str
    evidence: str = Field(description="Verbatim sentence from the source containing the number")


class RateResearchResult(BaseModel):
    """Wrapper the research crew must return."""

    rates: list[ResearchedRate] = Field(default_factory=list)


class ValidatedRate(BaseModel):
    """A rate after Python-side validation, carrying its own provenance."""

    key: str
    label: str
    value_pct: float
    origin: Literal["researched", "fallback"]
    as_of: str = "n/a"
    source_url: str = ""
    evidence: str = ""
    reject_reason: str = ""

    @property
    def rate(self) -> float:
        """Decimal form for maths, e.g. 0.071."""
        return self.value_pct / 100.0


class UserProfile(BaseModel):
    """Everything the deterministic engine needs. Missing values get documented defaults."""

    monthly_take_home: float = Field(gt=0, description="Net salary credited each month, in rupees")
    age: int = Field(default=30, ge=18, le=75)
    risk_profile: RiskProfile = "moderate"
    dependents: int = Field(default=0, ge=0)
    monthly_essentials: float | None = Field(
        default=None, description="Rent, food, bills, transport. Estimated if absent."
    )
    monthly_emi: float = Field(default=0, ge=0, description="Existing loan repayments")
    emergency_fund_now: float = Field(default=0, ge=0, description="Liquid savings already set aside")
    has_term_insurance: bool = False
    has_health_insurance: bool = False
    tax_regime: Literal["new"] = "new"

    @property
    def annual_take_home(self) -> float:
        return self.monthly_take_home * 12


class ProfileExtraction(BaseModel):
    """What the LLM is allowed to pull out of a chat message.

    Kept separate from `UserProfile` because every field is optional here -- the
    model reports only what the user actually said, and defaults are applied in
    Python rather than guessed by the LLM.
    """

    salary_amount: float | None = Field(default=None, description="Raw salary number in rupees")
    salary_is_annual: bool = Field(
        default=False, description="True only if the user clearly meant per year"
    )
    age: int | None = None
    risk_profile: RiskProfile | None = None
    dependents: int | None = None
    monthly_essentials: float | None = None
    monthly_emi: float | None = None
    emergency_fund_now: float | None = None
    has_term_insurance: bool | None = None
    has_health_insurance: bool | None = None


class Bucket(BaseModel):
    """One line of the recommended monthly allocation."""

    name: str
    instrument: str
    monthly_amount: float
    pct_of_income: float
    rate_key: str = ""
    expected_return_pct: float | None = None
    rationale: str = ""
    horizon_years: int | None = None


class Projection(BaseModel):
    name: str
    monthly_amount: float
    years: int
    expected_return_pct: float
    invested: float
    future_value: float
    gain: float
    real_future_value: float = Field(description="Future value in today's purchasing power")


class TaxBreakdown(BaseModel):
    fy: str
    source_url: str
    regime: str
    gross_annual: float
    standard_deduction: float
    taxable_income: float
    slab_tax: float
    rebate_87a: float
    surcharge: float
    cess: float
    total_tax: float
    effective_rate_pct: float


class FinancialPlan(BaseModel):
    """The complete deterministic output. Every number here came from Python."""

    profile: UserProfile
    assumed_essentials: float
    essentials_estimated: bool
    emergency_target: float
    emergency_gap: float
    investable_surplus: float
    buckets: list[Bucket]
    projections: list[Projection]
    equity_pct: int
    debt_pct: int
    gold_pct: int
    warnings: list[str] = Field(default_factory=list)
