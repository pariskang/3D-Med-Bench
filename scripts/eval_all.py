#!/usr/bin/env python3
"""
Evaluate all trace files against their rubrics. Produces stratified leaderboard.

Usage:
  python scripts/eval_all.py --traces traces/ --cases cases/ --report score,safety,tokens
  python scripts/eval_all.py --traces traces/ --cases cases/ --stratify difficulty,perception
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer()
console = Console()


@app.command()
def main(
    traces_dir: Path = typer.Option(Path("traces"), "--traces"),
    cases_dir: Path = typer.Option(Path("cases"), "--cases"),
    suite: str = typer.Option("v3", "--suite"),
    report: str = typer.Option("score,tokens,sim_cost,safety", "--report"),
    stratify: str = typer.Option("difficulty,perception", "--stratify"),
    output: Path = typer.Option(Path("leaderboard.json"), "--output"),
):
    """Evaluate all traces and produce a stratified leaderboard."""
    import anthropic
    from clinicraft.config import settings
    from clinicraft.judge.llm_judge import judge_encounter
    from clinicraft.judge.scorer import compute_score
    from clinicraft.schemas.rubric import Rubric

    trace_files = list(traces_dir.glob("**/*.json"))
    console.print(f"\n[bold]CliniCraft-Bench Evaluation[/bold]")
    console.print(f"  Traces: {len(trace_files)}")
    console.print(f"  Suite:  {suite}\n")

    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    async def _evaluate_all():
        scorecards = []
        for trace_path in trace_files:
            trace = json.loads(trace_path.read_text())
            case_id = trace.get("case_id", "")
            model_id = trace.get("model_id", "unknown")

            # Find corresponding case directory
            case_dirs = list(cases_dir.glob(f"**/{case_id}"))
            if not case_dirs:
                console.print(f"[yellow]Case dir not found for {case_id}, skipping[/yellow]")
                continue
            case_dir = case_dirs[0]

            rubric_path = case_dir / "tests" / "rubric.json"
            gtg_path = case_dir / "ground_truth_graph.json"
            if not rubric_path.exists() or not gtg_path.exists():
                continue

            try:
                verdict = await judge_encounter(
                    trace_path, rubric_path, gtg_path, model_id, client
                )
                rubric = Rubric.model_validate_json(rubric_path.read_text())
                card = compute_score(verdict, rubric, trace)
                scorecards.append(card)
            except Exception as e:
                console.print(f"[red]Error evaluating {case_id}: {e}[/red]")

        return scorecards

    cards = asyncio.run(_evaluate_all())

    if not cards:
        console.print("[yellow]No scores computed.[/yellow]")
        return

    # Aggregate results
    leaderboard = _build_leaderboard(cards, stratify.split(","))

    output.write_text(json.dumps(leaderboard, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[green]Leaderboard saved: {output}[/green]\n")

    _print_summary_table(cards, report.split(","))


def _build_leaderboard(cards, stratify_keys: list[str]) -> dict:
    from clinicraft.judge.scorer import ScoreCard

    models: dict[str, list] = {}
    for c in cards:
        models.setdefault(c.model_id, []).append(c)

    results = []
    for model_id, model_cards in sorted(models.items()):
        n = len(model_cards)
        avg = lambda attr: sum(getattr(c, attr) for c in model_cards) / n
        entry = {
            "model": model_id,
            "n_cases": n,
            "final": round(avg("final"), 2),
            "C1": round(avg("C1"), 2),
            "C2": round(avg("C2"), 2),
            "C3": round(avg("C3"), 2),
            "C4": round(avg("C4"), 2),
            "C6": round(avg("C6"), 2),
            "safety_veto_rate": round(sum(c.safety_veto for c in model_cards) / n, 3),
            "avg_tokens": round(avg("tokens_used")),
            "avg_tests": round(avg("tests_ordered"), 1),
            "avg_cost_cny": round(avg("sim_cost_cny"), 1),
        }
        results.append(entry)

    results.sort(key=lambda x: x["final"], reverse=True)
    return {
        "suite": "v3",
        "total_cases": len(cards),
        "leaderboard": results,
    }


def _print_summary_table(cards, report_cols: list[str]) -> None:
    from clinicraft.judge.scorer import ScoreCard

    table = Table(title="CliniCraft-Bench v3 Leaderboard")
    table.add_column("Model", style="bold")
    table.add_column("N", justify="right")
    table.add_column("Final", justify="right", style="cyan")
    table.add_column("C1", justify="right")
    table.add_column("C2", justify="right")
    table.add_column("C3", justify="right")
    table.add_column("C4", justify="right")
    table.add_column("C6", justify="right")
    if "safety" in report_cols:
        table.add_column("Safety✗%", justify="right", style="red")
    if "tokens" in report_cols:
        table.add_column("Tokens", justify="right")

    models: dict[str, list] = {}
    for c in cards:
        models.setdefault(c.model_id, []).append(c)

    for model_id, mc in sorted(models.items(), key=lambda x: -sum(c.final for c in x[1]) / len(x[1])):
        n = len(mc)
        avg = lambda a: round(sum(getattr(c, a) for c in mc) / n, 1)
        row = [
            model_id[:30], str(n),
            str(avg("final")), str(avg("C1")), str(avg("C2")),
            str(avg("C3")), str(avg("C4")), str(avg("C6")),
        ]
        if "safety" in report_cols:
            veto_pct = round(100 * sum(c.safety_veto for c in mc) / n, 1)
            row.append(f"{veto_pct}%")
        if "tokens" in report_cols:
            row.append(str(int(avg("tokens_used"))))
        table.add_row(*row)

    console.print(table)


if __name__ == "__main__":
    app()
