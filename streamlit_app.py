"""Streamlit web UI for the Financial Advisor Agent.

Run locally:  streamlit run streamlit_app.py
Deploy free:  push to GitHub → connect on share.streamlit.io
"""

from __future__ import annotations

import streamlit as st

from config import require_keys
from finance import format_inr
from graph import AdvisorState, build_graph
from schemas import FinancialPlan, TaxBreakdown, ValidatedRate

st.set_page_config(
    page_title="Financial Advisor Agent",
    page_icon="💰",
    layout="wide",
)

# ── Session state ──────────────────────────────────────────────────────────
if "state" not in st.session_state:
    st.session_state.state: AdvisorState = {}
if "app" not in st.session_state:
    st.session_state.app = None
if "messages" not in st.session_state:
    st.session_state.messages: list[dict] = []


def get_app():
    if st.session_state.app is None:
        st.session_state.app = build_graph()
    return st.session_state.app


# ── Rendering helpers ──────────────────────────────────────────────────────
def render_plan(plan: FinancialPlan) -> None:
    st.subheader("Recommended Monthly Allocation")

    rows = []
    for b in plan.buckets:
        rows.append({
            "Bucket": b.name,
            "Instrument": b.instrument,
            "Monthly": format_inr(b.monthly_amount),
            "% Income": f"{b.pct_of_income}%",
            "Exp. Return": f"{b.expected_return_pct}%" if b.expected_return_pct else "-",
        })
    total = sum(b.monthly_amount for b in plan.buckets)
    rows.append({
        "Bucket": "**Total**",
        "Instrument": "",
        "Monthly": f"**{format_inr(total)}**",
        "% Income": f"**{total / plan.profile.monthly_take_home * 100:.0f}%**",
        "Exp. Return": "",
    })
    st.table(rows)

    st.caption(
        f"Target mix: {plan.equity_pct}% equity / {plan.debt_pct}% debt / "
        f"{plan.gold_pct}% gold  ·  Investable surplus: "
        f"{format_inr(plan.investable_surplus)}/month"
    )


def render_projections(plan: FinancialPlan) -> None:
    if not plan.projections:
        return
    st.subheader("Growth Projection")
    rows = []
    for p in plan.projections:
        rows.append({
            "Horizon": p.name,
            "You Invest": format_inr(p.invested),
            "Projected Value": format_inr(p.future_value),
            "Growth": format_inr(p.gain),
            "In Today's Money": format_inr(p.real_future_value),
        })
    st.table(rows)
    st.caption(
        f"Blended expected return: {plan.projections[0].expected_return_pct}% p.a.  ·  "
        "Projections use researched historical rates, not a forecast."
    )


def render_tax(tax: TaxBreakdown) -> None:
    st.subheader(f"Income Tax on {format_inr(tax.gross_annual)}/year")
    cols = st.columns(4)
    cols[0].metric("Taxable Income", format_inr(tax.taxable_income))
    cols[1].metric("Slab Tax", format_inr(tax.slab_tax))
    cols[2].metric("87A Rebate", format_inr(tax.rebate_87a))
    cols[3].metric("Total Tax", format_inr(tax.total_tax), f"{tax.effective_rate_pct}% effective")
    st.caption(f"{tax.fy}  ·  {tax.regime}  ·  Std deduction {format_inr(tax.standard_deduction)}")


def render_sources(rates: dict[str, ValidatedRate]) -> None:
    for rate in rates.values():
        if rate.origin == "researched":
            st.markdown(
                f"**{rate.label}: {rate.value_pct}%** · :green[verified] · "
                f"as of {rate.as_of}"
            )
            st.caption(f'"{rate.evidence}"  —  [{rate.source_url}]({rate.source_url})')
        else:
            st.markdown(
                f"**{rate.label}: {rate.value_pct}%** · :orange[fallback]"
            )
            st.caption(f"Rejected: {rate.reject_reason}")


# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Financial Advisor Agent")
    st.caption(
        "Live rates researched by a CrewAI crew. "
        "All arithmetic computed in deterministic Python."
    )
    st.divider()

    if st.button("Reset conversation", use_container_width=True):
        st.session_state.state = {}
        st.session_state.messages = []
        st.session_state.app = None
        st.rerun()

    if st.session_state.state.get("rates"):
        with st.expander("Rate provenance (sources)"):
            render_sources(st.session_state.state["rates"])

    profile = st.session_state.state.get("profile")
    if profile:
        with st.expander("Current profile"):
            st.json(profile.model_dump())

    st.divider()
    st.caption(
        "Educational tool, not SEBI-registered investment advice. "
        "Verify all figures independently before acting."
    )


# ── Chat history ───────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("plan"):
            render_plan(msg["plan"])
            render_projections(msg["plan"])
        if msg.get("tax"):
            render_tax(msg["tax"])


# ── User input ─────────────────────────────────────────────────────────────
if prompt := st.chat_input("Tell me your salary, e.g. 'my salary is 1.2 lakhs, I'm 27'"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            require_keys()
        except SystemExit as exc:
            st.error(str(exc))
            st.stop()

        app = get_app()
        state = st.session_state.state

        fresh: AdvisorState = {
            "message": prompt,
            "profile": state.get("profile"),
            "rates": state.get("rates", {}),
            "plan": state.get("plan"),
            "tax": state.get("tax"),
            "attempts": 0,
            "failed_keys": [],
            "notes": [],
        }

        with st.spinner("Researching live rates and computing your plan..."):
            result: AdvisorState = app.invoke(fresh)

        for note in result.get("notes", []):
            st.caption(f"_{note}_")

        plan = result.get("plan")
        tax = result.get("tax")
        answer = result.get("answer", "")

        if plan and plan is not state.get("plan"):
            render_plan(plan)
            render_projections(plan)
            if tax:
                render_tax(tax)

        st.markdown(answer)

        st.session_state.state = result
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "plan": plan if plan and plan is not state.get("plan") else None,
            "tax": tax if plan and plan is not state.get("plan") else None,
        })

    st.rerun()
