"""Chat entry point.

Usage:
    python main.py                          # interactive chat
    python main.py "I take home 1.2 lakhs"  # single question, then exit
"""

from __future__ import annotations

import sys

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table

from config import require_keys
from finance import format_inr
from advisor_graph import AdvisorState, build_graph
from schemas import FinancialPlan, TaxBreakdown, ValidatedRate

console = Console()

BANNER = """[bold]Financial Advisor Agent[/bold]
Live rates researched by a CrewAI crew, all arithmetic computed in Python.

Tell me your salary to begin, e.g. [italic]"my salary is 1.2 lakhs, I'm 27, no loans"[/italic]
Commands: [cyan]/sources[/cyan]  [cyan]/profile[/cyan]  [cyan]/reset[/cyan]  [cyan]/quit[/cyan]

[dim]Educational tool, not SEBI-registered investment advice.[/dim]"""


def render_plan(plan: FinancialPlan) -> None:
    table = Table(title="Recommended monthly allocation", header_style="bold", expand=True)
    table.add_column("Bucket")
    table.add_column("Instrument", overflow="fold")
    table.add_column("Monthly", justify="right")
    table.add_column("% income", justify="right")
    table.add_column("Exp. return", justify="right")

    for bucket in plan.buckets:
        rate = f"{bucket.expected_return_pct}%" if bucket.expected_return_pct else "-"
        table.add_row(
            bucket.name,
            bucket.instrument,
            format_inr(bucket.monthly_amount),
            f"{bucket.pct_of_income}%",
            rate,
        )
    total = sum(b.monthly_amount for b in plan.buckets)
    table.add_section()
    table.add_row(
        "[bold]Total[/bold]", "", f"[bold]{format_inr(total)}[/bold]",
        f"[bold]{total / plan.profile.monthly_take_home * 100:.0f}%[/bold]", "",
    )
    console.print(table)
    console.print(
        f"[dim]Target mix {plan.equity_pct}% equity / {plan.debt_pct}% debt / "
        f"{plan.gold_pct}% gold - investable surplus "
        f"{format_inr(plan.investable_surplus)}/month[/dim]"
    )


def render_projections(plan: FinancialPlan) -> None:
    if not plan.projections:
        return
    table = Table(title="Growth projection", header_style="bold", expand=True)
    table.add_column("Horizon")
    table.add_column("You invest", justify="right")
    table.add_column("Projected value", justify="right")
    table.add_column("Growth", justify="right")
    table.add_column("In today's money", justify="right")
    for proj in plan.projections:
        table.add_row(
            proj.name,
            format_inr(proj.invested),
            format_inr(proj.future_value),
            format_inr(proj.gain),
            format_inr(proj.real_future_value),
        )
    console.print(table)
    console.print(
        f"[dim]Blended expected return {plan.projections[0].expected_return_pct}% p.a. "
        "Projections are arithmetic on researched historical rates, not a forecast.[/dim]"
    )


def render_tax(tax: TaxBreakdown) -> None:
    console.print(
        Panel(
            f"Taxable income {format_inr(tax.taxable_income)} after the "
            f"{format_inr(tax.standard_deduction)} standard deduction.\n"
            f"Slab tax {format_inr(tax.slab_tax)} - 87A rebate {format_inr(tax.rebate_87a)} "
            f"+ cess {format_inr(tax.cess)} = [bold]{format_inr(tax.total_tax)}[/bold] "
            f"({tax.effective_rate_pct}% effective)\n"
            f"[dim]{tax.fy} - {tax.regime} - {tax.source_url}[/dim]",
            title=f"Income tax on {format_inr(tax.gross_annual)}/year",
            border_style="dim",
        )
    )


def render_sources(rates: dict[str, ValidatedRate]) -> None:
    """The audit trail: every rate, its status, and the sentence it came from."""
    console.print(Rule("[bold]Rate provenance[/bold]"))
    for rate in rates.values():
        if rate.origin == "researched":
            console.print(
                f"[green]verified[/green] [bold]{rate.label}: {rate.value_pct}%[/bold] "
                f"[dim](as of {rate.as_of})[/dim]"
            )
            console.print(f'  [italic dim]"{rate.evidence}"[/italic dim]')
            console.print(f"  [dim]{rate.source_url}[/dim]\n")
        else:
            console.print(
                f"[yellow]fallback[/yellow] [bold]{rate.label}: {rate.value_pct}%[/bold]"
            )
            console.print(f"  [dim]rejected because: {rate.reject_reason}[/dim]\n")


def run_turn(app, state: AdvisorState, message: str) -> AdvisorState:
    fresh: AdvisorState = {
        "message": message,
        "profile": state.get("profile"),
        "rates": state.get("rates", {}),
        "plan": state.get("plan"),
        "tax": state.get("tax"),
        "attempts": 0,
        "failed_keys": [],
        "notes": [],
    }

    had_plan = state.get("plan")
    with console.status("[dim]Thinking: extracting profile, researching live rates...[/dim]"):
        result: AdvisorState = app.invoke(fresh)

    for note in result.get("notes", []):
        console.print(f"[dim]. {note}[/dim]")

    plan = result.get("plan")
    if plan is not None and plan is not had_plan:
        console.print()
        render_plan(plan)
        render_projections(plan)
        if result.get("tax"):
            render_tax(result["tax"])

    console.print()
    console.print(Panel(Markdown(result.get("answer", "")), title="Advice", border_style="cyan"))
    console.print(
        "[dim]Educational information, not SEBI-registered investment advice. "
        "Run /sources to see where every rate came from.[/dim]"
    )
    return result


def main() -> None:
    require_keys()
    app = build_graph()
    state: AdvisorState = {}

    if len(sys.argv) > 1:
        try:
            run_turn(app, state, " ".join(sys.argv[1:]))
        except Exception as exc:  # noqa: BLE001 - report cleanly instead of a stack trace
            console.print(f"[red]Error:[/red] {type(exc).__name__}: {exc}")
            raise SystemExit(1) from exc
        return

    console.print(Panel(BANNER, border_style="cyan"))
    while True:
        try:
            message = console.input("\n[bold cyan]you >[/bold cyan] ").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]bye[/dim]")
            return

        if not message:
            continue
        lowered = message.lower()
        if lowered in ("/quit", "/exit", "quit", "exit"):
            console.print("[dim]bye[/dim]")
            return
        if lowered == "/reset":
            state = {}
            console.print("[dim]Context cleared.[/dim]")
            continue
        if lowered == "/sources":
            if state.get("rates"):
                render_sources(state["rates"])
            else:
                console.print("[dim]No rates researched yet.[/dim]")
            continue
        if lowered == "/profile":
            profile = state.get("profile")
            console.print(profile.model_dump() if profile else "[dim]No profile yet.[/dim]")
            continue

        try:
            state = run_turn(app, state, message)
        except Exception as exc:  # noqa: BLE001 - keep the REPL alive on API errors
            console.print(f"[red]Error:[/red] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
