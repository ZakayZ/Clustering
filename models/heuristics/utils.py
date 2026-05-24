from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch

from cluster_energy import partition_loss_numpy

@dataclass(frozen=True)
class EventBaseline:
    node_labels: np.ndarray
    loss: float
    partition: list[list[int]]

def make_event_baseline(
    pos: np.ndarray,
    mom: np.ndarray,
    is_proton: np.ndarray,
    partition: list[list[int]],
) -> EventBaseline:
    node_lab = np.zeros((pos.shape[0],), dtype=np.int32)
    for ci, clus in enumerate(partition):
        node_lab[np.asarray(clus, dtype=np.int64)] = int(ci)
    loss = float(partition_loss_numpy(pos, mom, is_proton, partition))
    return EventBaseline(node_labels=node_lab, loss=loss, partition=partition)

def partition_within_cluster_edge_pairs(partition: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    ei: list[int] = []
    ej: list[int] = []
    for c in partition:
        if len(c) < 2:
            continue
        for i, j in combinations(sorted(int(x) for x in c), 2):
            ei.append(i)
            ej.append(j)
    return np.asarray(ei, dtype=np.int64), np.asarray(ej, dtype=np.int64)

def edge_pair_baseline_targets(
    node_labels: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
) -> torch.Tensor:
    tgt = (node_labels[edge_i] == node_labels[edge_j]).astype(np.float32)
    return torch.tensor(tgt, dtype=torch.float32)
