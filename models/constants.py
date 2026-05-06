"""Shared numeric / typing constants for graph policy and environment."""

from __future__ import annotations

from typing import Literal

# Physics uses **MeV** internally (:func:`cluster_energy.partition_loss_numpy`, env rewards).
# Use ``MEV_PER_GEV`` only to convert MeV → GeV for tqdm/readouts/plots (monitoring).
MEV_PER_GEV = 1000.0

# 7-D scaled phase-space for nodes & edges (see ``AffinityGraphConfig`` feature scales).
# Edge attr: 3×7 (Δφ, φ_i, φ_j) + 2 (|Δr|/dr_cut_scale, |Δk|/dk_cut_scale) = 23.
EDGE_PHYS_DIM = 23

# GAT node x: scaled phase-space (7) + is_proton (1) = 8.
GAT_NODE_IN_DIM = 8

GraphKind = Literal["knn", "radius", "full"]
