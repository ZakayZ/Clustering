"""Graph construction, connectivity helpers, and ``AffinityGraphEnv``."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph
from torch_geometric.data import Data

from cluster_energy import NucleonCloud, cluster_energy, partition_loss_numpy

from models.graph import GraphBuilder
from models.heuristics.constants import HBARC_MEV_FM, K_CUT_FM_INV, R_CUT_FM
from models.heuristics.protocol import BaselineModel


def labels_to_partition(labels: np.ndarray) -> list[list[int]]:
    by_lbl = defaultdict[int, list[int]](list)
    for i, lab in enumerate(labels.astype(int).tolist()):
        by_lbl[int(lab)].append(i)
    return [by_lbl[k] for k in sorted(by_lbl.keys())]


def partition_to_labels(n: int, part: list[list[int]]) -> np.ndarray:
    """Dense labels ``0 .. n_clusters-1`` from an explicit partition."""
    lab = np.zeros(int(n), dtype=np.int32)
    for ci, c in enumerate(part):
        for j in c:
            lab[int(j)] = int(ci)
    return lab


def dissolve_unfavorable_clusters(
    pos: np.ndarray,
    mom: np.ndarray,
    isp: np.ndarray,
    partition: list[list[int]],
    *,
    energy_threshold_mev: float,
) -> list[list[int]]:
    """Split multi-nucleon clusters with ``total_energy >= threshold`` into singletons (MeV)."""
    cloud = NucleonCloud.from_numpy(pos, mom, isp)
    out: list[list[int]] = []
    thr = float(energy_threshold_mev)
    for c in partition:
        if len(c) <= 1:
            out.append(list(c))
            continue
        te = cluster_energy(cloud, c).total_energy
        if te >= thr:
            for i in sorted(c):
                out.append([i])
        else:
            out.append(sorted(c))
    out.sort(key=len, reverse=True)
    return out


def cluster_labels_from_edges(
    n: int,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_on: np.ndarray,
) -> np.ndarray:
    """Connected components on the subgraph of **on** edges (labels ``0 .. n_comp-1``).

    Uses ``scipy.sparse.csgraph.connected_components`` (CSR adjacency, undirected).
    """
    on = np.asarray(edge_on, dtype=bool).reshape(-1)
    ei = np.asarray(edge_i, dtype=np.int64)[on]
    ej = np.asarray(edge_j, dtype=np.int64)[on]

    _, labels = csgraph.connected_components(
        csr_matrix((np.ones(ei.size, dtype=np.int8), (ei, ej)), shape=(n, n)),
        directed=False,
        return_labels=True,
    )
    return labels.astype(np.int32, copy=False)


class AffinityGraphEnv:
    """``reset`` stores ``mom``, ``isp`` as given and normalizes ``pos`` to ``(N, 4)`` ``(t,x,y,z)``.

    If ``pos`` is ``(N, 3)`` spatial fm only, a column of zeros is prepended as coordinate time.

    ``reset`` fills ``_baseline_node_labels`` / ``_baseline_loss`` from the required
    :class:`~models.heuristics.protocol.BaselineModel` (e.g.
    :class:`~models.heuristics.coalescence.CoalescenceHeuristicModel` with cut-like params).
    Optional ``event_index`` is forwarded to the baseline so indexed caches stay aligned with data order.

    Graph topology comes from a :class:`~models.graph.GraphBuilder` (kNN / radius / full).
    Before topology, ``reset`` converts spatial ``r3`` (fm) and wavevector ``k3`` (fm^-1)
    to cut units using ``dr_cut_scale_fm`` / ``dk_cut_scale_fm_inv``, then passes those arrays to
    the builder (so builders never divide by ``R_CUT``/``K_CUT`` themselves).
    Node and edge features use ``feat_scale_*`` and cut scales below (defaults match the prior
    standalone config dataclass that was merged into this constructor).

    The returned ``Data`` has node features ``x``, ``edge_index`` / ``edge_attr`` from the builder,
    plus ``edge_pair_*`` and ``policy_edge_idx`` for the undirected policy edge list (``src < dst``).

    After ``reset``, ``graph`` is the single ``Data`` passed to the policy."""

    def __init__(
        self,
        graph_builder: GraphBuilder,
        baseline: BaselineModel,
        *,
        feat_scale_r_fm: float = 50.0,
        feat_scale_e: float = 2000.0,
        feat_scale_k_fm_inv: float = 5.0,
        dr_cut_scale_fm: float = R_CUT_FM,
        dk_cut_scale_fm_inv: float = K_CUT_FM_INV,
        cluster_dissolve_energy_threshold_mev: float | None = None,
    ) -> None:
        self._graph_builder = graph_builder
        self.baseline = baseline
        self._feat_scale_r_fm = float(feat_scale_r_fm)
        self._feat_scale_e = float(feat_scale_e)
        self._feat_scale_k_fm_inv = float(feat_scale_k_fm_inv)
        self._dr_cut_scale_fm = float(dr_cut_scale_fm)
        self._dk_cut_scale_fm_inv = float(dk_cut_scale_fm_inv)
        self._cluster_dissolve_energy_threshold_mev = cluster_dissolve_energy_threshold_mev
        self.pos = np.zeros((0, 4), dtype=np.float64)
        self.mom = np.zeros((0, 4), dtype=np.float64)
        self.isp = np.zeros((0,), dtype=bool)
        self.graph = Data()
        self._baseline_node_labels = np.zeros((0,), dtype=np.int32)
        self._baseline_loss = 0.0
        self._partition_loss = 0.0

    @property
    def n_real_edges(self) -> int:
        return int(self.graph.edge_pair_i.shape[0])

    def reset(
        self,
        pos: np.ndarray,
        mom: np.ndarray,
        is_proton: np.ndarray,
        *,
        event_index: int | None = None,
    ) -> Data:
        pos_a = np.asarray(pos, dtype=np.float64)
        if pos_a.shape[1] == 3:
            pos_geo = np.concatenate(
                [np.zeros((pos_a.shape[0], 1), dtype=np.float64), pos_a], axis=1
            )
        else:
            pos_geo = pos_a
        k3 = mom[:, 1:] / HBARC_MEV_FM
        r3 = pos_geo[:, 1:4]
        e = mom[:, :1]

        self.pos = pos
        self.mom = mom
        self.isp = is_proton

        sr = self._feat_scale_r_fm
        se = self._feat_scale_e
        sk = self._feat_scale_k_fm_inv
        sdr = self._dr_cut_scale_fm
        sdk = self._dk_cut_scale_fm_inv

        r3_cut = np.asarray(r3, dtype=np.float64) / sdr
        k3_cut = np.asarray(k3, dtype=np.float64) / sdk
        self.graph = self._graph_builder.build(r3_cut, k3_cut)

        # Scaled phase-space for nodes & edges (t dropped — always 0 for 3-D datasets).
        phase_space = np.concatenate(
            [r3 / sr, e / se, k3 / sk],
            axis=1,
            dtype=np.float32,
        )  # shape (N, 7)
        self.graph.x = torch.from_numpy(np.concatenate(
            [phase_space, is_proton[:, None]],
            axis=1,
            dtype=np.float32,
        ))  # shape (N, 8)
        phase_space_t = torch.from_numpy(phase_space)
        i, j = self.graph.edge_index
        delta = phase_space_t[i] - phase_space_t[j]
        # |Δr|, |Δk| in cut units (= 1 at baseline boundary): cols 0:3 = Δ(r/sr), 4:7 = Δ(k/sk).
        dr_cut = delta[:, 0:3].norm(dim=1, keepdim=True) * (sr / sdr)
        dk_cut = delta[:, 4:7].norm(dim=1, keepdim=True) * (sk / sdk)
        self.graph.edge_attr = torch.cat(
            [delta, phase_space_t[i], phase_space_t[j], dr_cut, dk_cut],
            dim=1,
        )  # shape (E_dir, 23)
        bl = self.baseline(self.pos, self.mom, self.isp, event_index=event_index)
        self._baseline_node_labels = bl.node_labels.astype(np.int32)
        self._baseline_loss = float(bl.loss)
        return self.graph

    def physics_for_edge_mask(self, edge_on: torch.Tensor) -> tuple[float, np.ndarray]:
        n = self.pos.shape[0]
        on = edge_on.detach().numpy().astype(bool, copy=False).reshape(-1)
        labels_cc = cluster_labels_from_edges(
            n, self.graph.edge_pair_i.numpy(), self.graph.edge_pair_j.numpy(), on
        )
        part = labels_to_partition(labels_cc)
        thr = self._cluster_dissolve_energy_threshold_mev
        if thr is not None:
            part = dissolve_unfavorable_clusters(
                self.pos, self.mom, self.isp, part, energy_threshold_mev=float(thr)
            )
        labels = partition_to_labels(n, part)
        pl = float(partition_loss_numpy(self.pos, self.mom, self.isp, part))
        self._partition_loss = pl
        return pl, labels
