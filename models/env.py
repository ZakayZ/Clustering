"""Graph construction, connectivity helpers, and ``AffinityGraphEnv``."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse import csgraph
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch_geometric.transforms import KNNGraph
from torch_geometric.utils import to_undirected

from cluster_energy import NucleonCloud, cluster_energy, partition_loss_numpy

from models.affinity_graph_config import AffinityGraphConfig
from models.constants import GraphKind
from models.heuristics.constants import HBARC_MEV_FM, K_CUT_FM_INV, R_CUT_FM
from models.heuristics.protocol import BaselineModel


def _policy_edges_from_directed(ei: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split directed ``edge_index`` into undirected policy pairs (``src < dst``) and index mask."""
    src, dst = ei[0], ei[1]
    keep = src < dst
    return src[keep], dst[keep], torch.nonzero(keep, as_tuple=True)[0].long()


def _build_knn_graph(pos: np.ndarray, mom_phys_or_k: np.ndarray, transform: KNNGraph) -> Data:
    """kNN in ``(r, k)`` (6-D): ``r`` fm; second arg is either ``mom`` ``(N,4)`` MeV/c or ``k`` ``(N,3)`` fm⁻¹."""
    m = np.asarray(mom_phys_or_k, dtype=np.float64)
    if m.shape[1] == 4:
        k = m[:, 1:4] / HBARC_MEV_FM
    else:
        k = m
    six = np.concatenate([np.asarray(pos, dtype=np.float64), k], axis=1).astype(np.float32)
    d = transform(Data(pos=torch.tensor(six)))
    ei = d.edge_index
    pi, pj, pidx = _policy_edges_from_directed(ei)
    d.edge_pair_i = pi
    d.edge_pair_j = pj
    d.policy_edge_idx = pidx
    return d


def _build_radius_graph(r3: np.ndarray, k3: np.ndarray, radius_norm: float) -> Data:
    """Radius edges in normalized ``(r/R_cut, k/K_cut)`` 6-D Euclidean space.

    ``radius_norm ≈ 1`` connects pairs separated by ~one baseline gate scale in mixed units.
    Falls back to kNN ``k=3`` if no edges (too-small radius on sparse clouds).
    """
    n = int(r3.shape[0])
    x = np.concatenate(
        [np.asarray(r3, dtype=np.float64) / R_CUT_FM, np.asarray(k3, dtype=np.float64) / float(K_CUT_FM_INV)],
        axis=1,
    ).astype(np.float32)
    pos_t = torch.from_numpy(x)
    ei = pyg_nn.radius_graph(pos_t, r=float(radius_norm), loop=False)
    if ei.numel() == 0 or n <= 1:
        fallback = KNNGraph(k=min(3, max(n - 1, 1)), loop=False, force_undirected=True)
        return _build_knn_graph(r3, k3, fallback)
    ei = to_undirected(ei, num_nodes=n)
    d = Data(edge_index=ei)
    pi, pj, pidx = _policy_edges_from_directed(ei)
    d.edge_pair_i = pi
    d.edge_pair_j = pj
    d.policy_edge_idx = pidx
    return d


def _build_complete_graph(n: int) -> Data:
    """All ordered pairs ``i → j``, ``i ≠ j`` (bidirectional), for full message passing."""
    if n <= 1:
        return Data(edge_index=torch.zeros((2, 0), dtype=torch.long), edge_pair_i=torch.zeros(0, dtype=torch.long), edge_pair_j=torch.zeros(0, dtype=torch.long), policy_edge_idx=torch.zeros(0, dtype=torch.long))
    rows: list[int] = []
    cols: list[int] = []
    for i in range(n):
        for j in range(n):
            if i != j:
                rows.append(i)
                cols.append(j)
    ei = torch.tensor([rows, cols], dtype=torch.long)
    d = Data(edge_index=ei)
    pi, pj, pidx = _policy_edges_from_directed(ei)
    d.edge_pair_i = pi
    d.edge_pair_j = pj
    d.policy_edge_idx = pidx
    return d


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

    The returned ``Data`` has node features ``x``, ``edge_index`` / ``edge_attr`` from
    kNN / radius / complete topology (one ``edge_attr`` row per directed edge), plus
    ``edge_pair_*`` and ``policy_edge_idx`` for the undirected policy edge list (``src < dst``).

    After ``reset``, ``graph`` is the single ``Data`` passed to the policy."""

    def __init__(self, cfg: AffinityGraphConfig, baseline: BaselineModel) -> None:
        self.cfg = cfg
        self.baseline = baseline
        self._knn = (
            KNNGraph(k=cfg.k_nn, loop=False, force_undirected=True)
            if cfg.graph_kind == "knn"
            else None
        )
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

    def reset(self, pos: np.ndarray, mom: np.ndarray, is_proton: np.ndarray) -> Data:
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

        if self.cfg.graph_kind == "knn":
            assert self._knn is not None
            self.graph = _build_knn_graph(r3, k3, self._knn)
        elif self.cfg.graph_kind == "radius":
            self.graph = _build_radius_graph(r3, k3, float(self.cfg.radius_norm))
        elif self.cfg.graph_kind == "full":
            self.graph = _build_complete_graph(int(r3.shape[0]))
        else:
            raise ValueError(f"unknown graph_kind {self.cfg.graph_kind!r}")

        sr = float(self.cfg.feat_scale_r_fm)
        se = float(self.cfg.feat_scale_e)
        sk = float(self.cfg.feat_scale_k_fm_inv)
        sdr = float(self.cfg.dr_cut_scale_fm)
        sdk = float(self.cfg.dk_cut_scale_fm_inv)

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
        bl = self.baseline(self.pos, self.mom, self.isp)
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
        thr = self.cfg.cluster_dissolve_energy_threshold_mev
        if thr is not None:
            part = dissolve_unfavorable_clusters(
                self.pos, self.mom, self.isp, part, energy_threshold_mev=float(thr)
            )
        labels = partition_to_labels(n, part)
        pl = float(partition_loss_numpy(self.pos, self.mom, self.isp, part))
        self._partition_loss = pl
        return pl, labels
