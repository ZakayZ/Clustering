"""Re-export baseline API from :mod:`models.heuristics` for stable imports."""

from __future__ import annotations

from models.heuristics.annealing import CCLAnnealParams
from models.heuristics.coalescence import CoalescenceBaselineParams
from models.heuristics.constants import (
    HBARC_MEV_FM,
    K_CUT_FM_INV,
    Q_CUT_GEVC,
    Q_CUT_MEVC,
    R_CUT_FM,
)
from models.heuristics.coalescence import baseline_clusters_numpy, fast_coalescence_partition
from models.heuristics.utils import (
    EventBaseline,
    edge_pair_baseline_targets,
    make_event_baseline,
)

__all__ = [
    "HBARC_MEV_FM",
    "K_CUT_FM_INV",
    "Q_CUT_GEVC",
    "Q_CUT_MEVC",
    "R_CUT_FM",
    "CCLAnnealParams",
    "CoalescenceBaselineParams",
    "EventBaseline",
    "baseline_clusters_numpy",
    "edge_pair_baseline_targets",
    "fast_coalescence_partition",
    "make_event_baseline",
]
