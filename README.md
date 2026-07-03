# CliniCraft-Bench v3.0

**具身式复杂病例问诊评测基准** — an execution-grounded, embodied-3D, physiologically-grounded clinical evaluation benchmark that turns a corpus of real complex cases into playable, contamination-free evaluation scripts.

> Can a model perform expert-level, embodied clinical reasoning — perceiving, examining, and managing a physiologically-grounded 3D patient — across a large corpus of real complex cases?

This repository implements the CliniCraft-Bench v3.0 protocol: a 7-stage pipeline that ingests raw case text (`.txt`) and produces playable case scripts, plus a typed observation–action environment, a six-dimensional judge, and oracle/nop calibration anchors.

---

## Quickstart

```bash
pip install -e .
cp .env.example .env          # set ANTHROPIC_API_KEY

# Ingest a directory of .txt case files → playable cases/
python scripts/ingest_corpus.py --in /secure/cases_20k --out cases/ --n 500 --strata balanced

# Calibration bounds (oracle needs no API key; nop/doctor invoke the patient LLM)
./scripts/run.sh --case cases/cardiology/CASE_001 --agent oracle
./scripts/run.sh --case cases/cardiology/CASE_001 --agent nop

# Run a doctor model under the 3D active-perception setting
./scripts/run_doctor.sh --model claude-opus-4-8 --perception frame_stream --case cases/cardiology/CASE_001

# Stratified leaderboard
./scripts/eval_all.sh --traces traces/ --cases cases/ --report score,tokens,sim_cost,safety
```

---

## Architecture

```
L6 Governance   GB/T 39725/42460 de-id · PIPL · AIGC labelling      governance/
L5 Evaluation   6-dim metrics + hidden-rubric multimodal judge      clinicraft/judge/
L4 Interaction  typed observation-action loop + consequence engine  clinicraft/environment/
L3 Embodiment   avatar + Sign Rendering Library + scene renderer     clinicraft/render/
L2 Patient Host persona + atomic-facts hallucination guard           clinicraft/patient/
L1 Physiology   Pulse (live) ⊕ Findings Library (scripted)           clinicraft/physio/
L0 Case scripts 7-stage ingestion pipeline                           clinicraft/pipeline/
```

**7-stage pipeline** (`clinicraft/pipeline/stage1..7`):
ingest → de-identify → ground-truth graph → physiological grounding → embodiment → package → QC.

**Six evaluation dimensions** (`clinicraft/judge/`):
C1 diagnostic correctness · C2 reasoning quality · C3 3D interaction & perception ·
C4 process & management · C5 safety (hard-veto) · C6 calibration.

---

## Framework selections (2024–2026 maturity)

| Component | Choice | Rationale |
|---|---|---|
| Extraction | Anthropic SDK `tool_use` + Pydantic v2 | Guaranteed structure over free-form Chinese clinical text; beats rule-based cTAKES/HanLP for multi-section recall |
| De-identification | Presidio + custom Chinese PHI recognizers | philter is English-only; per-field scrubbing (not serialized JSON) prevents fail-open |
| Physiology | Pulse Engine SDK + scripted fallback | Successor to BioGears; auto-degrades to `MockPulseClient` when SDK absent |
| Rendering (Tier A) | Godot 4 headless + stub renderer | Open, lightweight, headless-native; stub keeps the loop runnable pre-integration |
| Orchestration | Prefect 2.x (+ plain-async fallback) | Async-native; more Pythonic than Airflow, better than Kedro for dynamic LLM chains |
| Ontologies | obonet (HPO), phenopackets, pandas/LOINC | obonet is the most stable OBO parser; phenopackets for rare-disease phenotypes |

---

## Implementation status (honest)

This is an actively developed benchmark. The table distinguishes **functional**
(runs end-to-end, tested), **stub** (runnable placeholder, not
publication-grade), and **planned** (integration seam exists, not yet built).

| Capability | Status | Notes |
|---|---|---|
| Pydantic schemas (case / GTG / interaction / rubric / pack) | ✅ functional | 24 unit tests |
| Stage 1 ingestion (txt → ClinicalCase) | ✅ functional | needs `ANTHROPIC_API_KEY` |
| Stage 2 de-identification (GB/T 42460) | ✅ functional | per-field scrub, **fails closed**; Presidio NER upgrade planned |
| Stage 3–7 (GTG → grounding → embody → pack → QC) | ✅ functional | GTG is LLM-drafted; expert validation is manual (see below) |
| Typed observation–action loop (§5) | ✅ functional | oracle runs end-to-end offline |
| Six-dimension scorer + hard-veto (§7) | ✅ functional | renormalised; veto covers auto + LLM + multimodal items |
| Oracle / nop calibration anchors (§8) | ✅ functional | oracle exercises C1–C4/C6 |
| Consequence engine (treatment → vitals) | ⚠️ stub | linear delta table, not real physiology |
| Physiology (Pulse Engine) | ⚠️ stub | `MockPulseClient` unless SDK installed & wired |
| 3D rendering (frame_stream) | ⚠️ stub | `StubSceneRenderer` draws labelled placeholder PNGs. **Photoreal Godot/UE5 is planned** — do not publish frame_stream C3 results from stub renders |
| Multimodal judge (C3) | ✅ functional | scores real frames; scores 0 when no image frame is available |
| Expert double-validation of GTG (Stage 3) | ⛔ planned | `validated=False` until a human annotation workflow lands |
| §2.2 cognitive-error taxonomy (DEER/Graber) | ⛔ planned | not yet computed |
| §2.5 real ECE + 10-run consistency | ⛔ planned | current calibration check is a heuristic |
| TCM 望闻问切 track (§12) | ⚠️ partial | actions routed to exam resolver; sign library seeded, dedicated 辨证 rubric planned |

See the protocol for the full specification. The planned items (renderer, Pulse
wiring, annotation UI, ECE/consistency) are the next milestones (§11 P1–P2).

---

## Development

```bash
pip install -e ".[dev]"
python -m pytest tests/ -q          # 24 tests, no API key required
ruff check clinicraft/
```

License: Apache-2.0.
