#!/usr/bin/env python3
"""
Stage-3 expert annotation CLI — GTG validation + inter-rater reliability.

Workflow:
  # 1. generate blank review forms for two specialists + an arbitrator
  python scripts/annotate_gtg.py create-tasks --cases cases/ \
      --out annotations/ --annotators dr_wang,dr_li,dr_arb

  # 2. (experts edit annotations/<case_id>/<annotator>.yaml)

  # 3. inter-rater reliability across all annotated cases (κ ≥ 0.8 gate)
  python scripts/annotate_gtg.py irr --annotations annotations/ --threshold 0.8

  # 4. merge → validated GTG written back into the case directory
  python scripts/annotate_gtg.py finalize --case cases/cardiology/CASE_001 \
      --annotations annotations/
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from rich.console import Console
from rich.table import Table

from clinicraft.annotation.irr import interpret_kappa
from clinicraft.annotation.workflow import create_tasks, finalize_case, run_irr

app = typer.Typer(help="Expert GTG annotation & inter-rater reliability")
console = Console()


@app.command("create-tasks")
def cmd_create_tasks(
    cases: Path = typer.Option(..., "--cases"),
    out: Path = typer.Option(Path("annotations"), "--out"),
    annotators: str = typer.Option("dr_1,dr_2", "--annotators",
                                   help="comma-separated ids; use *_arb for arbitrator"),
):
    """Emit a blank YAML review form per (case, annotator)."""
    ids = [a.strip() for a in annotators.split(",") if a.strip()]
    written = create_tasks(cases, out, ids)
    console.print(f"[green]Wrote {len(written)} forms → {out}[/green]")
    console.print(f"  Annotators: {', '.join(ids)}")
    console.print("  Note: rename an annotator with role 'arbitrator' inside the YAML "
                  "to designate the tie-breaker.")


@app.command("irr")
def cmd_irr(
    annotations: Path = typer.Option(..., "--annotations"),
    threshold: float = typer.Option(0.8, "--threshold"),
    output: Path = typer.Option(None, "--output"),
):
    """Compute inter-rater reliability (κ) across annotated cases."""
    report = run_irr(annotations, threshold=threshold)
    if report.n_cases == 0:
        console.print("[yellow]No cases with ≥2 expert annotations found.[/yellow]")
        raise typer.Exit(1)

    table = Table(title=f"Inter-rater reliability  (n={report.n_cases} cases, "
                        f"{report.n_raters} raters, gate κ≥{threshold})")
    table.add_column("Variable", style="bold")
    table.add_column("κ", justify="right")
    table.add_column("Method")
    table.add_column("Agreement")
    table.add_column("Gate", justify="center")
    for v in report.variables:
        gate = "[green]PASS[/green]" if v.passes_gate else "[red]FAIL[/red]"
        table.add_row(v.variable, f"{v.kappa:.3f}", v.method, v.interpretation, gate)
    console.print(table)

    verdict = "[green]PASS[/green]" if report.overall_pass else "[red]FAIL[/red]"
    console.print(f"\nMean κ = [cyan]{report.mean_kappa:.3f}[/cyan] "
                  f"({interpret_kappa(report.mean_kappa)}) — overall gate: {verdict}")

    if output:
        output.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
                          encoding="utf-8")
        console.print(f"[green]Saved: {output}[/green]")


@app.command("finalize")
def cmd_finalize(
    case: Path = typer.Option(..., "--case"),
    annotations: Path = typer.Option(..., "--annotations"),
    dry_run: bool = typer.Option(False, "--dry-run"),
):
    """Merge annotations → validated GTG (written back unless --dry-run)."""
    result = finalize_case(case, annotations, write=not dry_run)
    status = "[green]VALIDATED[/green]" if result.validated else "[yellow]UNRESOLVED[/yellow]"
    console.print(f"\nCase {result.case_id}: {status}")
    console.print(f"  Experts: {', '.join(result.experts)}")
    if result.arbitrator:
        console.print(f"  Arbitrator: {result.arbitrator}")
    if result.disagreements:
        console.print(f"  Disagreements ({len(result.disagreements)}):")
        for d in result.disagreements:
            resolved = f" → resolved by {d.resolved_by}" if d.resolved_by else " [red](unresolved)[/red]"
            console.print(f"    - {d.field_name}: {d.verdicts}{resolved}")
    if result.unresolved:
        console.print(f"  [red]Unresolved fields: {result.unresolved}[/red]")
    if result.apply_errors:
        console.print(f"  [red]Apply errors: {result.apply_errors}[/red]")
    if dry_run:
        console.print("  [dim](dry-run — nothing written)[/dim]")


if __name__ == "__main__":
    app()
