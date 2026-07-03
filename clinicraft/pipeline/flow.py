"""
Prefect 2.x orchestration flow for the 7-stage pipeline.

Why Prefect over Airflow/Kedro:
- Python-native async tasks (no DAG YAML needed)
- Built-in retry, caching, and concurrency controls
- State persistence enables resume after failure
- Kedro is excellent for ML pipelines but less suited for dynamic async LLM chains

Usage:
  prefect deploy clinicraft/pipeline/flow.py --name clinicraft-ingest
  prefect run deployment clinicraft-ingest
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import anthropic
from loguru import logger

try:
    from prefect import flow, task, get_run_logger
    from prefect.concurrency.asyncio import concurrency
    HAS_PREFECT = True
except ImportError:
    HAS_PREFECT = False
    # Fallback: bare decorators that do nothing
    def flow(fn=None, **kw):
        return fn if fn else lambda f: f

    def task(fn=None, **kw):
        return fn if fn else lambda f: f

    def get_run_logger():
        return logger

    class _FakeConcurrency:
        async def __aenter__(self): return self
        async def __aexit__(self, *a): pass

    async def concurrency(name, occupy=1):
        return _FakeConcurrency()


from clinicraft.config import settings
from clinicraft.pipeline.stage1_ingest import ingest_file
from clinicraft.pipeline.stage2_deid import deid_case
from clinicraft.pipeline.stage3_gtg import build_gtg
from clinicraft.pipeline.stage4_physio import ground_case
from clinicraft.pipeline.stage5_embody import embody_case
from clinicraft.pipeline.stage6_pack import pack_case
from clinicraft.pipeline.stage7_qc import qc_case_directory


@task(retries=2, retry_delay_seconds=5, name="stage1-ingest")
async def task_ingest(txt_path: str, client: anthropic.AsyncAnthropic):
    return await ingest_file(Path(txt_path), client=client)


@task(name="stage2-deid")
def task_deid(case_json: str):
    from clinicraft.schemas.clinical_case import ClinicalCase
    case = ClinicalCase.model_validate_json(case_json)
    clean, report = deid_case(case)
    return clean.model_dump_json(), report.model_dump()


@task(retries=2, retry_delay_seconds=5, name="stage3-gtg")
async def task_gtg(case_json: str, client: anthropic.AsyncAnthropic):
    from clinicraft.schemas.clinical_case import ClinicalCase
    case = ClinicalCase.model_validate_json(case_json)
    return await build_gtg(case, client=client)


@task(name="stage4-physio")
async def task_physio(gtg_json: str):
    from clinicraft.schemas.ground_truth import GroundTruthGraph
    gtg = GroundTruthGraph.model_validate_json(gtg_json)
    return await ground_case(gtg)


@task(name="stage5-embody")
def task_embody(case_json: str, gtg_json: str):
    from clinicraft.schemas.clinical_case import ClinicalCase
    from clinicraft.schemas.ground_truth import GroundTruthGraph
    case = ClinicalCase.model_validate_json(case_json)
    gtg = GroundTruthGraph.model_validate_json(gtg_json)
    return embody_case(case, gtg)


@task(name="stage6-pack")
def task_pack(case_json: str, gtg_json: str, physio_result: dict, embody_result: dict):
    from clinicraft.schemas.clinical_case import ClinicalCase
    from clinicraft.schemas.ground_truth import GroundTruthGraph
    case = ClinicalCase.model_validate_json(case_json)
    gtg = GroundTruthGraph.model_validate_json(gtg_json)
    out = pack_case(case, gtg, physio_result, embody_result)
    return str(out)


@task(name="stage7-qc")
def task_qc(case_dir: str):
    report = qc_case_directory(Path(case_dir))
    return report.model_dump()


@flow(name="clinicraft-ingest-single", log_prints=True)
async def ingest_single_flow(txt_path: str) -> dict:
    """Full 7-stage pipeline for one txt file."""
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    case = await task_ingest(txt_path, client)
    case_json = case.model_dump_json()

    clean_json, deid_report = task_deid(case_json)

    gtg = await task_gtg(clean_json, client)
    gtg_json = gtg.model_dump_json()

    physio = await task_physio(gtg_json)
    embody = task_embody(clean_json, gtg_json)
    case_dir = task_pack(clean_json, gtg_json, physio, embody)
    qc = task_qc(case_dir)

    return {"case_dir": case_dir, "qc": qc, "deid_report": deid_report}


@flow(name="clinicraft-ingest-batch", log_prints=True)
async def ingest_batch_flow(
    txt_dir: str,
    n: int = 500,
    strata: str = "balanced",
) -> list[dict]:
    """Batch ingestion of up to n txt files from a directory."""
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    paths = sorted(Path(txt_dir).glob("**/*.txt"))[:n]
    logger.info(f"Ingesting {len(paths)} files from {txt_dir}")

    results = []
    sem = asyncio.Semaphore(settings.max_concurrent)

    async def _one(p: Path) -> dict:
        async with sem:
            return await ingest_single_flow(str(p))

    results = await asyncio.gather(*[_one(p) for p in paths], return_exceptions=True)
    ok = [r for r in results if isinstance(r, dict)]
    logger.success(f"Batch complete: {len(ok)}/{len(paths)} succeeded")
    return ok


# ---------------------------------------------------------------------------
# Simple sync runner (no Prefect required)
# ---------------------------------------------------------------------------

async def run_pipeline_simple(txt_path: Path) -> dict:
    """Non-Prefect sync runner for development / testing."""
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)

    case = await ingest_file(txt_path, client=client)
    clean, deid_report = deid_case(case)
    gtg = await build_gtg(clean, client=client)
    physio = await ground_case(gtg)
    embody = embody_case(clean, gtg)
    case_dir = pack_case(clean, gtg, physio, embody)
    qc = qc_case_directory(case_dir)

    return {"case_dir": str(case_dir), "qc": qc.model_dump()}
