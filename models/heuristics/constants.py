"""Cut / phase-space scales for clustering heuristics.

This package only depends on :mod:`cluster_energy` from outside ``heuristics``; scales live here
rather than in :mod:`models.constants`.
"""

from __future__ import annotations

MEV_PER_GEV = 1000.0

R_CUT_FM = 7.0
Q_CUT_GEVC = 0.12
Q_CUT_MEVC = Q_CUT_GEVC * MEV_PER_GEV
HBARC_MEV_FM = 197.327
K_CUT_FM_INV = Q_CUT_MEVC / HBARC_MEV_FM
