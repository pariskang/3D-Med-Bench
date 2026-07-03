#!/usr/bin/env python3
"""
§2.5 consistency harness — run one case N times (temperature > 0) and report
diagnostic stability ("competence without consistency").

Usage:
  python scripts/run_consistency.py --case cases/cardiology/CASE_001 \
      --model claude-opus-4-8 --runs 10 --perception structured_only
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
    case_dir: Path = typer.Option(..., "--case", "-p"),
    model: str = typer.Option("claude-opus-4-8", "--model"),
    runs: int = typer.Option(10, "--runs"),
    perception: str = typer.Option("structured_only", "--perception"),
    output: Path = typer.Option(Path("consistency.json"), "--output"),
):
    """Run a case N times and report diagnostic consistency."""
    from clinicraft.metrics.consistency import diagnostic_consistency, flip_rate, extract_final_dx

    # Reuse the episode runner from run_doctor.
    from run_doctor import _run_episode  # type: ignore[import]
    from clinicraft.schemas.case_pack import CasePack
    from clinicraft.schemas.ground_truth import GroundTruthGraph
    from clinicraft.schemas.interaction import PerceptionMode

    pack = CasePack.load(case_dir)
    gtg = GroundTruthGraph.model_validate_json((case_dir / "ground_truth_graph.json").read_text())
    pmode = PerceptionMode(perception)

    console.print(f"\n[bold]Consistency run[/bold]  case={pack.case_id} model={model} runs={runs}")

    async def _all() -> list[dict]:
        traces = []
        for i in range(runs):
            console.print(f"  run {i + 1}/{runs} …")
            traces.append(await _run_episode(pack, gtg, model, pmode, case_dir, seed=1000 + i))
        return traces

    traces = asyncio.run(_all())
    final_dxs = [extract_final_dx(t) or "(none)" for t in traces]
    rep = diagnostic_consistency(final_dxs)

    table = Table(title=f"Consistency — {pack.case_id}")
    table.add_column("metric"); table.add_column("value", justify="right")
    table.add_row("runs", str(rep.n_runs))
    table.add_row("distinct diagnoses", str(rep.distinct_dx))
    table.add_row("modal diagnosis", str(rep.modal_dx))
    table.add_row("modal agreement", f"{rep.modal_agreement:.2f}")
    table.add_row("normalised entropy", f"{rep.normalised_entropy:.2f}")
    table.add_row("flip rate", f"{flip_rate(final_dxs):.2f}")
    table.add_row("consistency score", f"{rep.consistency_score:.1f}/100")
    console.print(table)
    console.print(f"  distribution: {rep.distribution}")

    result = {
        "case_id": pack.case_id, "model": model, "runs": runs,
        "ground_truth": gtg.final_dx,
        "final_diagnoses": final_dxs,
        "report": {
            "modal_dx": rep.modal_dx, "modal_agreement": rep.modal_agreement,
            "normalised_entropy": rep.normalised_entropy,
            "consistency_score": rep.consistency_score,
            "distribution": rep.distribution,
        },
    }
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    console.print(f"[green]Saved: {output}[/green]")


if __name__ == "__main__":
    app()
