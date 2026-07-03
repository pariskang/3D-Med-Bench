#!/usr/bin/env python3
"""
Stage 1-7 batch ingestion CLI.
Converts a directory of .txt case files → playable CliniCraft case directories.

Usage:
  python scripts/ingest_corpus.py --in /secure/cases_20k --out cases/ --n 500
  python scripts/ingest_corpus.py --in ./my_cases --out cases/ --n 10 --dry-run

Requires: ANTHROPIC_API_KEY set (via .env or environment)
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import typer
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

from clinicraft.config import settings
from clinicraft.pipeline.flow import run_pipeline_simple
from clinicraft.pipeline.stage7_qc import generate_release_manifest

app = typer.Typer(name="clinicraft-ingest", help="Ingest .txt case files into CliniCraft-Bench")
console = Console()


def _discover_txt_files(
    in_dir: Path, n: int, strata: str, specialty_filter: str | None
) -> list[Path]:
    """Discover and optionally filter/sample txt files."""
    all_files = sorted(in_dir.glob("**/*.txt"))
    if specialty_filter:
        all_files = [f for f in all_files if specialty_filter.lower() in f.parts[-2].lower()]

    if strata == "balanced" and n < len(all_files):
        # Try to pick from multiple subdirectories (specialties)
        by_dir: dict[str, list[Path]] = {}
        for f in all_files:
            d = str(f.parent)
            by_dir.setdefault(d, []).append(f)
        result: list[Path] = []
        per_dir = max(1, n // len(by_dir))
        for paths in by_dir.values():
            result.extend(paths[:per_dir])
        return result[:n]

    return all_files[:n]


@app.command()
def main(
    input_dir: Path = typer.Option(..., "--in", help="Directory containing .txt case files"),
    output_dir: Path = typer.Option(Path("cases"), "--out", help="Output cases directory"),
    n: int = typer.Option(500, "--n", help="Max number of files to process"),
    strata: str = typer.Option("balanced", "--strata", help="Sampling strategy: balanced|sequential"),
    specialty: str | None = typer.Option(None, "--specialty", help="Filter by specialty subfolder name"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Discover files but don't process"),
    manifest_out: Path | None = typer.Option(None, "--manifest", help="Write release manifest JSON"),
    max_concurrent: int = typer.Option(settings.max_concurrent, "--concurrent"),
    model: str = typer.Option(settings.llm_model, "--model"),
):
    """Ingest .txt medical case files → CliniCraft-Bench playable case directories."""

    if not input_dir.exists():
        console.print(f"[red]Input directory not found: {input_dir}[/red]")
        raise typer.Exit(1)

    if not settings.anthropic_api_key:
        console.print("[red]ANTHROPIC_API_KEY not set. Check .env file.[/red]")
        raise typer.Exit(1)

    files = _discover_txt_files(input_dir, n, strata, specialty)
    console.print(f"\n[bold]CliniCraft-Bench Ingestion Pipeline[/bold]")
    console.print(f"  Input:       {input_dir}")
    console.print(f"  Output:      {output_dir}")
    console.print(f"  Files found: [cyan]{len(files)}[/cyan]")
    console.print(f"  Model:       {model}")
    console.print(f"  Concurrent:  {max_concurrent}\n")

    if dry_run:
        console.print("[yellow]Dry run — listing files only:[/yellow]")
        for f in files[:20]:
            console.print(f"  {f}")
        if len(files) > 20:
            console.print(f"  ... and {len(files) - 20} more")
        return

    # Process files
    results = asyncio.run(_process_all(files, output_dir, max_concurrent, model))

    # Summary table
    ok = [r for r in results if r.get("qc", {}).get("passed")]
    fail = [r for r in results if not r.get("qc", {}).get("passed")]

    table = Table(title="Ingestion Results")
    table.add_column("Status", style="bold")
    table.add_column("Count", justify="right")
    table.add_row("[green]Passed QC[/green]", str(len(ok)))
    table.add_row("[red]Failed QC[/red]", str(len(fail)))
    table.add_row("Total", str(len(results)))
    console.print(table)

    if fail:
        console.print("\n[yellow]Failed cases:[/yellow]")
        for r in fail[:5]:
            issues = r.get("qc", {}).get("issues", [])
            console.print(f"  {r.get('case_dir')}: {issues}")

    # Release manifest
    if manifest_out or ok:
        case_dirs = [Path(r["case_dir"]) for r in ok]
        manifest = generate_release_manifest(case_dirs)
        out_path = manifest_out or output_dir / "manifest.json"
        out_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        console.print(f"\n[green]Manifest written: {out_path}[/green]")
        console.print(f"  Total cases: {manifest['total']}")
        console.print(f"  By difficulty: {manifest['by_difficulty']}")
        console.print(f"  By specialty: {manifest['by_specialty']}")


async def _process_all(
    files: list[Path],
    output_dir: Path,
    max_concurrent: int,
    model: str,
) -> list[dict]:
    sem = asyncio.Semaphore(max_concurrent)
    results = []

    async def _one(path: Path) -> dict:
        async with sem:
            try:
                result = await run_pipeline_simple(path)
                logger.info(f"✓ {path.name} → {result['case_dir']}")
                return result
            except Exception as e:
                logger.error(f"✗ {path.name}: {e}")
                return {"case_dir": "", "qc": {"passed": False, "issues": [str(e)]}}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    ) as progress:
        task = progress.add_task("Processing cases...", total=len(files))
        tasks = []
        for f in files:
            t = asyncio.create_task(_one(f))
            t.add_done_callback(lambda _: progress.advance(task))
            tasks.append(t)
        results = list(await asyncio.gather(*tasks))

    return results


if __name__ == "__main__":
    app()
