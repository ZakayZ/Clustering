import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.transforms import KNNGraph

from models.graph.base import GraphBuilder
from models.graph.utils import policy_edges_from_directed

class KNNGraphBuilder(GraphBuilder):
    def __init__(
        self,
        k: int = 6,
        *,
        loop: bool = False,
        force_undirected: bool = True,
    ) -> None:
        self._transform = KNNGraph(k=k, loop=loop, force_undirected=force_undirected)

    def build(self, r3: np.ndarray, k3: np.ndarray) -> Data:
        d = self._transform(Data(pos=torch.tensor(np.concatenate([r3, k3], axis=1, dtype=np.float32))))
        ei = d.edge_index
        pi, pj, pidx = policy_edges_from_directed(ei)
        d.edge_pair_i = pi
        d.edge_pair_j = pj
        d.policy_edge_idx = pidx
        return d
