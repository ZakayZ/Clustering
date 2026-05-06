"""Graph layout hyperparameters for :class:`~models.env.AffinityGraphEnv`."""

from __future__ import annotations

from dataclasses import dataclass

from models.constants import GraphKind
from models.heuristics.constants import K_CUT_FM_INV, R_CUT_FM


@dataclass
class AffinityGraphConfig:
    """Graph construction (kNN / radius / full); independent of cluster baseline choice.

    ``feat_scale_*`` rescale raw phase-space before nodes/edges enter the policy.
    ``dr_cut_scale_fm`` / ``dk_cut_scale_fm_inv`` define cut units for the two |Δ·| scalars
    appended to ``edge_attr``.  BatchNorm1d on edges sees ~O(10³) directed edges per event,
    so edge statistics are already rich; the scales still matter for gradient conditioning.
    """

    k_nn: int = 6
    graph_kind: GraphKind = "knn"
    # radius_graph in normalized ``(r/R_cut, k/K_cut)`` 6-D space (only ``graph_kind=='radius'``).
    radius_norm: float = 1.2
    # Typical UrQMD spreads: z-positions ±~170 fm, k ~ few fm⁻¹, E ~ GeV — tune via these only.
    feat_scale_r_fm: float = 50.0
    # Normalizes ``E`` (MeV) for node/edge features; default ~2 GeV equivalent.
    feat_scale_e: float = 2000.0
    feat_scale_k_fm_inv: float = 5.0
    # Phase-space edge scalars: denominators for |Δr|, |Δk| (defaults = spatial–momentum cut units).
    dr_cut_scale_fm: float = R_CUT_FM
    dk_cut_scale_fm_inv: float = K_CUT_FM_INV
    # After connected-components from the edge mask, optional post-processing: split any
    # multi-nucleon cluster whose :func:`~cluster_energy.cluster_energy` ``total_energy``
    # is **>=** this threshold (MeV) into singletons — same rule as
    # ``CoalescenceBaselineParams.dissolve_energy_threshold`` when ``drop_unfavorable_clusters``.
    # ``None`` disables (raw CC partition is scored as-is).
    cluster_dissolve_energy_threshold_mev: float | None = None
