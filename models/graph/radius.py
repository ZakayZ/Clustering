import numpy as np
import torch
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from models.graph.base import GraphBuilder
from models.graph.utils import policy_edges_from_directed

class RadiusGraphBuilder(GraphBuilder):
    def __init__(
        self,
        radius_norm: float = 1.2,
    ) -> None:
        self._radius_norm = float(radius_norm)

    def build(self, r3: np.ndarray, k3: np.ndarray) -> Data:
        n = int(r3.shape[0])
        x = np.concatenate(
            [np.asarray(r3, dtype=np.float64), np.asarray(k3, dtype=np.float64)],
            axis=1,
        ).astype(np.float32)
        pos_t = torch.from_numpy(x)
        ei = pyg_nn.radius_graph(pos_t, r=float(self._radius_norm), loop=False)
        ei = to_undirected(ei, num_nodes=n)
        d = Data(edge_index=ei)
        pi, pj, pidx = policy_edges_from_directed(ei)
        d.edge_pair_i = pi
        d.edge_pair_j = pj
        d.policy_edge_idx = pidx
        return d
