"""Spatial–momentum cut graph, partitions, and :class:`CoalescenceHeuristicModel`.

The **cut baseline** (lab ``|Δr|`` + ``|Δk|`` gates, momentum always on, no dissolve) is
:class:`CoalescenceHeuristicModel` with default :class:`CoalescenceBaselineParams` — same graph as
:func:`baseline_clusters_numpy` on a full event.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from cluster_energy import NucleonCloud, cluster_energy

from models.heuristics.constants import HBARC_MEV_FM, Q_CUT_MEVC, R_CUT_FM
from models.heuristics.utils import EventBaseline, make_event_baseline
from models.heuristics.annealing import CCLAnnealParams



def _pos3_stack(pos: np.ndarray) -> np.ndarray:
    p = np.asarray(pos, dtype=np.float64)
    return p[:, 1:4] if p.shape[-1] == 4 else p


def _components_to_partition(
    idx: np.ndarray,
    n_local: int,
    labels: np.ndarray,
) -> list[list[int]]:
    """Map connected-component labels on local 0..n_local-1 to global nucleon indices."""
    by: dict[int, list[int]] = defaultdict(list)
    for k in range(n_local):
        by[int(labels[k])].append(int(idx[k]))
    comps = [sorted(by[g]) for g in sorted(by.keys())]
    comps.sort(key=len, reverse=True)
    return comps


def _cut_adjacency_edges(
    pos3: np.ndarray,
    k3: np.ndarray,
    *,
    r_cut_fm: float,
    use_momentum_gate: bool,
    q_cut_momentum: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return undirected CSR edge list (both directions) for pairs passing the gates."""
    n = int(pos3.shape[0])
    if n <= 1:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    dr = pos3[:, None, :] - pos3[None, :, :]
    dist_r = np.sqrt(np.sum(dr * dr, axis=2), dtype=np.float64)
    mask = dist_r < float(r_cut_fm)
    if use_momentum_gate:
        k_cut = float(q_cut_momentum) / HBARC_MEV_FM
        dk = k3[:, None, :] - k3[None, :, :]
        dist_k = np.sqrt(np.sum(dk * dk, axis=2), dtype=np.float64)
        mask &= dist_k < k_cut
    iu, ju = np.triu_indices(n, k=1)
    keep = mask[iu, ju]
    ei0 = iu[keep]
    ej0 = ju[keep]
    if ei0.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    ei = np.concatenate([ei0, ej0])
    ej = np.concatenate([ej0, ei0])
    return ei.astype(np.int64), ej.astype(np.int64)


def baseline_clusters_numpy(
    pos: np.ndarray,
    mom: np.ndarray,
    indices: list[int],
    r_cut_fm: float,
    q_cut_momentum: float,
) -> list[list[int]]:
    """Cut-based clusters: neighbors if ``|Δr| < r_cut`` and ``|Δk| < k_cut``.

    ``pos`` is ``(N, 3)`` or ``(N, 4)`` (only spatial columns are used).
    ``mom`` is ``(N, 4)`` with ``(E, px, py, pz)`` in **MeV/c**;
    the cut uses ``k = p_spatial / ħc`` (fm⁻¹).

    Uses NumPy broadcast + ``scipy.sparse.csgraph.connected_components``.
    """
    pos = np.asarray(pos, dtype=np.float64)
    pos3 = _pos3_stack(pos)
    idx = np.asarray(indices, dtype=np.int64)
    n = int(idx.size)
    if n == 0:
        return []
    if n == 1:
        return [[int(idx[0])]]

    mom = np.asarray(mom, dtype=np.float64)
    p = pos3[idx]
    k3 = mom[idx, 1:4] / HBARC_MEV_FM
    ei, ej = _cut_adjacency_edges(
        p,
        k3,
        r_cut_fm=r_cut_fm,
        use_momentum_gate=True,
        q_cut_momentum=float(q_cut_momentum),
    )
    graph = csr_matrix(
        (np.ones(int(ei.size), dtype=np.int8), (ei, ej)),
        shape=(n, n),
    )
    _, lab = connected_components(graph, directed=False, return_labels=True)
    return _components_to_partition(idx, n, lab)


def fast_coalescence_partition(
    pos: np.ndarray,
    mom: np.ndarray,
    is_proton: np.ndarray,
    *,
    radius_fm: float,
    use_momentum_gate: bool = True,
    q_cut_momentum: float = Q_CUT_MEVC,
    drop_unfavorable_clusters: bool = False,
    dissolve_energy_threshold: float = 0.0,
) -> list[list[int]]:
    """Fast surrogate: same graph builder as cut with configurable gates + optional UrQMD dissolve."""
    pos = np.asarray(pos, dtype=np.float64)
    mom = np.asarray(mom, dtype=np.float64)
    isp = np.asarray(is_proton, dtype=bool)
    n_ev = int(pos.shape[0])
    if n_ev == 0:
        return []
    idx = np.arange(n_ev, dtype=np.int64)
    pos3 = _pos3_stack(pos)
    k3 = mom[:, 1:4] / HBARC_MEV_FM
    ei, ej = _cut_adjacency_edges(
        pos3,
        k3,
        r_cut_fm=float(radius_fm),
        use_momentum_gate=bool(use_momentum_gate),
        q_cut_momentum=float(q_cut_momentum),
    )
    graph = csr_matrix(
        (np.ones(int(ei.size), dtype=np.int8), (ei, ej)),
        shape=(n_ev, n_ev),
    )
    _, lab = connected_components(graph, directed=False, return_labels=True)
    part = _components_to_partition(idx, n_ev, lab)

    if not drop_unfavorable_clusters:
        return part

    cloud = NucleonCloud.from_numpy(pos, mom, isp)
    out: list[list[int]] = []
    thr = float(dissolve_energy_threshold)
    for c in part:
        if len(c) <= 1:
            out.append(c)
            continue
        te = cluster_energy(cloud, c).total_energy
        if te >= thr:
            for i in c:
                out.append([i])
        else:
            out.append(sorted(c))
    out.sort(key=len, reverse=True)
    return out


@dataclass(frozen=True)
class CoalescenceBaselineParams:
    """Hyperparameters for :class:`CoalescenceHeuristicModel`.

    **Defaults** (``use_momentum_gate=True``, ``drop_unfavorable_clusters=False``) match the
    former **cut** baseline on full events — same graph as :func:`baseline_clusters_numpy`.

    Other settings turn off the momentum gate and/or dissolve clusters via
    :func:`~cluster_energy.cluster_energy` ``total_energy``.
    """

    radius_fm: float = R_CUT_FM
    use_momentum_gate: bool = True
    q_cut_momentum: float = Q_CUT_MEVC
    drop_unfavorable_clusters: bool = False
    dissolve_energy_threshold: float = 0.0
    rng_seed: int = 0


@dataclass
class CoalescenceHeuristicModel:
    """Vector cut graph + optional UrQMD dissolve (same as legacy ``method='coalescence'`` / cut)."""

    params: CoalescenceBaselineParams = field(default_factory=CoalescenceBaselineParams)

    def __call__(
        self,
        pos: np.ndarray,
        mom: np.ndarray,
        is_proton: np.ndarray,
        *,
        event_index: int | None = None,
    ) -> EventBaseline:
        c = self.params
        part_b = fast_coalescence_partition(
            pos,
            mom,
            is_proton,
            radius_fm=float(c.radius_fm),
            use_momentum_gate=bool(c.use_momentum_gate),
            q_cut_momentum=float(c.q_cut_momentum),
            drop_unfavorable_clusters=bool(c.drop_unfavorable_clusters),
            dissolve_energy_threshold=float(c.dissolve_energy_threshold),
        )
        return make_event_baseline(pos, mom, is_proton, part_b)


#: Back-compat alias — same as :class:`~models.heuristics.annealing.CCLAnnealParams` **defaults**
#: (tuned via ``scripts/search_coalescence_anneal.py``, see
#: ``clustering/benchmarks/coalesce_ccl_anneal_search.json`` and
#: ``coalesce_ccl_anneal_search_uncapped_40ev52trials.json`` for long uncapped search).
RECOMMENDED_CCL_ANNEAL_FOR_DEFAULT_COALESCENCE = CCLAnnealParams()
