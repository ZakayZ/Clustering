"""CCL-style MST / percolation (pair-COM proximity). See :class:`MSTModel` for tuned defaults."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from models.baseline import (
    EventBaseline,
    make_event_baseline,
)
from models.heuristics.coalescence import _components_to_partition, _pos3_stack

# Tuned on ``urqmd_nucleons_1k`` (see ``scripts/search_mst_negative_loss.py``): negative mean partition loss.
MST_DEFAULT_R_CUT_FM = 4.5
MST_DEFAULT_P_CUT_MEVC = 40.0


def _boost_spatial_batch(
    t: np.ndarray,
    r: np.ndarray,
    beta: np.ndarray,
    *,
    beta_sq_eps: float = 1e-24,
) -> np.ndarray:
    """
    Boost spatial coordinates ``r`` together with time ``t`` using isotropic ``beta``.

    Same Lorentz transformation as for a 4-vector ``(t, r)`` with boost velocity ``beta``.
    Shapes: ``t`` (...,), ``r`` (..., 3), ``beta`` (..., 3); broadcasting on leading dims.
    """
    beta_sq = np.sum(beta * beta, axis=-1)
    small = beta_sq < beta_sq_eps
    eps = 1e-30
    safe_bs = np.maximum(beta_sq, eps)
    gamma = 1.0 / np.sqrt(np.maximum(1.0 - beta_sq, eps))
    beta_dot_r = np.sum(beta * r, axis=-1)
    inv_bs = (gamma - 1.0) / safe_bs
    r_prime = (
        r
        + inv_bs[..., None] * beta_dot_r[..., None] * beta
        - (gamma * t)[..., None] * beta
    )
    return np.where(small[..., None], r, r_prime)


def _pair_com_separation_fm(pos3: np.ndarray, t: np.ndarray, p4: np.ndarray) -> np.ndarray:
    """Pairwise spatial separation in fm in the pair COM frame; shape ``(N, N)`` symmetric."""
    n = int(pos3.shape[0])
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)
    e = p4[:, 0]
    pm = p4[:, 1:4]
    t_inv = np.maximum(e[:, None] + e[None, :], 1e-30)
    beta = (pm[:, None, :] + pm[None, :, :]) / t_inv[..., None]
    ri = pos3[:, None, :]
    rj = pos3[None, :, :]
    ti = t[:, None]
    tj = t[None, :]
    ri_p = _boost_spatial_batch(ti, ri, beta)
    rj_p = _boost_spatial_batch(tj, rj, beta)
    d = ri_p - rj_p
    return np.sqrt(np.maximum(np.sum(d * d, axis=-1), 0.0))


def _cluster_passes_momentum_gate(
    mom_loc: np.ndarray,
    member_idx: np.ndarray,
    p_cut_mevc: float,
) -> bool:
    """CCL-style: COM from summed lab 4-momenta; each constituent must have ``|p| < p_cut`` in that frame."""
    if member_idx.size <= 1:
        return True
    p4 = mom_loc[np.asarray(member_idx, dtype=np.int64)]
    s = np.sum(p4, axis=0)
    e_tot = float(s[0])
    if e_tot <= 0.0:
        return False
    beta = s[1:4] / e_tot
    pc = float(p_cut_mevc)
    for row in p4:
        p_boosted = _boost_spatial_batch(
            np.asarray(row[0], dtype=np.float64),
            row[1:4].astype(np.float64, copy=False),
            beta,
        )
        if float(np.linalg.norm(p_boosted)) >= pc:
            return False
    return True


def _dsu_find(parent: np.ndarray, x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[parent[x]]]
        x = int(parent[x])
    return int(x)


def _mst_partition_dsu_gated(
    dist_fm: np.ndarray,
    mom_loc: np.ndarray,
    *,
    r_cut_fm: float,
    check_pair_momentum: bool,
    check_merge_momentum: bool,
    p_cut_mevc: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return local labels ``0..n-1`` after DSU with CCL-like optional momentum gates."""
    n = int(dist_fm.shape[0])
    parent = np.arange(n, dtype=np.int64)
    iu, ju = np.triu_indices(n, k=1)
    mask = (dist_fm[iu, ju] <= float(r_cut_fm)) & (iu != ju)
    pairs = np.stack([iu[mask], ju[mask]], axis=1).astype(np.int64, copy=False)
    if pairs.shape[0] > 1:
        order = rng.permutation(int(pairs.shape[0]))
        pairs = pairs[order]

    def unite(a: int, b: int) -> None:
        ra = _dsu_find(parent, a)
        rb = _dsu_find(parent, b)
        if ra == rb:
            return
        if check_pair_momentum:
            if not _cluster_passes_momentum_gate(
                mom_loc, np.asarray([a, b], dtype=np.int64), p_cut_mevc
            ):
                return
        if check_merge_momentum:
            members = np.array(
                [i for i in range(n) if _dsu_find(parent, i) in (ra, rb)],
                dtype=np.int64,
            )
            if not _cluster_passes_momentum_gate(mom_loc, members, p_cut_mevc):
                return
        parent[rb] = ra

    for a, b in pairs:
        unite(int(a), int(b))

    roots = np.asarray([_dsu_find(parent, i) for i in range(n)], dtype=np.int64)
    _, lab = np.unique(roots, return_inverse=True)
    return lab.astype(np.int64, copy=False)


def mst_partition_numpy(
    pos: np.ndarray,
    mom: np.ndarray,
    indices: list[int],
    *,
    r_cut_fm: float = MST_DEFAULT_R_CUT_FM,
    check_pair_momentum: bool = False,
    check_merge_momentum: bool = True,
    p_cut_mevc: float = MST_DEFAULT_P_CUT_MEVC,
    rng_seed: int = 0,
) -> list[list[int]]:
    """
    CCL-style MST clusters: connect if pair separation in the pair COM frame ``<= r_cut_fm``.

    Default ``r_cut_fm``, ``p_cut_mevc``, and merge gate match :class:`MSTModel`.
    Set ``check_pair_momentum`` and ``check_merge_momentum`` to ``False`` for the fast
    connected-components path (no momentum gates).
    """
    pos0 = np.asarray(pos, dtype=np.float64)
    mom0 = np.asarray(mom, dtype=np.float64)
    idx = np.asarray(indices, dtype=np.int64)
    n = int(idx.size)
    if n == 0:
        return []
    if n == 1:
        return [[int(idx[0])]]

    pos3_full = _pos3_stack(pos0)
    if pos0.shape[-1] == 4:
        t_full = pos0[:, 0].astype(np.float64, copy=False)
    else:
        t_full = np.zeros((pos0.shape[0],), dtype=np.float64)
    pos3 = pos3_full[idx]
    t = t_full[idx]
    mom_loc = mom0[idx].astype(np.float64, copy=False)

    p4 = mom_loc
    dist_fm = _pair_com_separation_fm(pos3, t, p4)
    np.fill_diagonal(dist_fm, np.inf)

    use_gates = bool(check_pair_momentum or check_merge_momentum)
    if not use_gates:
        prox = dist_fm <= float(r_cut_fm)
        np.fill_diagonal(prox, False)
        ei0, ej0 = np.nonzero(prox)
        if ei0.size == 0:
            ei = np.zeros(0, dtype=np.int64)
            ej = np.zeros(0, dtype=np.int64)
        else:
            ei = np.concatenate([ei0, ej0]).astype(np.int64)
            ej = np.concatenate([ej0, ei0]).astype(np.int64)
        graph = csr_matrix(
            (np.ones(int(ei.size), dtype=np.int8), (ei, ej)),
            shape=(n, n),
        )
        _, lab = connected_components(graph, directed=False, return_labels=True)
        return _components_to_partition(idx, n, lab)

    rng = np.random.default_rng(int(rng_seed))
    lab = _mst_partition_dsu_gated(
        dist_fm,
        mom_loc,
        r_cut_fm=float(r_cut_fm),
        check_pair_momentum=bool(check_pair_momentum),
        check_merge_momentum=bool(check_merge_momentum),
        p_cut_mevc=float(p_cut_mevc),
        rng=rng,
    )
    return _components_to_partition(idx, n, lab)


@dataclass
class MSTModel:
    """
    Pair-COM spatial proximity (CCL-style) with optional momentum gates.

    Defaults target **negative** mean :func:`~cluster_energy.partition_loss_numpy` on
    ``urqmd_nucleons_1k`` (seed 0 train/hold splits): merge gate with
    ``r_cut_fm`` and ``p_cut_mevc`` defaults: :data:`MST_DEFAULT_R_CUT_FM`,
    :data:`MST_DEFAULT_P_CUT_MEVC`. Re-tune with ``scripts/search_mst_negative_loss.py``.
    """

    r_cut_fm: float = MST_DEFAULT_R_CUT_FM
    check_pair_momentum: bool = False
    check_merge_momentum: bool = True
    p_cut_mevc: float = MST_DEFAULT_P_CUT_MEVC
    rng_seed: int = 0

    def __call__(
        self,
        pos: np.ndarray,
        mom: np.ndarray,
        is_proton: np.ndarray,
    ) -> EventBaseline:
        n_ev = int(pos.shape[0])
        part = mst_partition_numpy(
            pos,
            mom,
            list(range(n_ev)),
            r_cut_fm=float(self.r_cut_fm),
            check_pair_momentum=bool(self.check_pair_momentum),
            check_merge_momentum=bool(self.check_merge_momentum),
            p_cut_mevc=float(self.p_cut_mevc),
            rng_seed=int(self.rng_seed),
        )
        return make_event_baseline(pos, mom, is_proton, part)
