"""
Expert annotation workflow (§3 stage-3 validation).

Turns LLM-drafted Ground-Truth Graphs into expert-validated ground truth via a
double-blind ≥2-physician review + arbitration, with inter-rater reliability
(Cohen / weighted / Fleiss κ) reported against the protocol's κ≥0.8 gate.

- schema:    GTGAnnotation form (one expert's review of one case)
- irr:       Cohen κ, quadratic/linear weighted κ, Fleiss κ, Landis-Koch bands
- consensus: merge annotations → validated GTG + disagreement report
- workflow:  create tasks, load annotations, run IRR, finalize a case
"""

from clinicraft.annotation.schema import (
    GTGAnnotation, FieldJudgment, StrataJudgment, Verdict, AnnotationTask,
)
from clinicraft.annotation.irr import (
    cohen_kappa, weighted_cohen_kappa, fleiss_kappa, interpret_kappa,
    compute_irr, IRRReport,
)
from clinicraft.annotation.consensus import (
    merge_annotations, ConsensusResult, DisagreementItem,
)

__all__ = [
    "GTGAnnotation", "FieldJudgment", "StrataJudgment", "Verdict", "AnnotationTask",
    "cohen_kappa", "weighted_cohen_kappa", "fleiss_kappa", "interpret_kappa",
    "compute_irr", "IRRReport",
    "merge_annotations", "ConsensusResult", "DisagreementItem",
]
