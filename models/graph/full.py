"""Fully connected (ordered pairs) graph topology."""

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse

from models.graph.base import GraphBuilder
from models.graph.utils import policy_edges_from_directed


class FullGraphBuilder(GraphBuilder):
    """All ordered pairs ``i → j``, ``i ≠ j`` (bidirectional) for full message passing."""

    def build(self, r3: np.ndarray, _: np.ndarray) -> Data:
        n = int(r3.shape[0])
        adj = torch.ones((n, n), dtype=torch.float32)
        adj.fill_diagonal_(0)
        ei, _ = dense_to_sparse(adj)
        d = Data(edge_index=ei)
        pi, pj, pidx = policy_edges_from_directed(ei)
        d.edge_pair_i = pi
        d.edge_pair_j = pj
        d.policy_edge_idx = pidx
        return d
