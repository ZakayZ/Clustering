"""Shared baseline outputs: :class:`EventBaseline`, :func:`make_event_baseline`, edge targets."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from cluster_energy import partition_loss_numpy


@dataclass(frozen=True)
class EventBaseline:
    """Per-event baseline: node-wise cluster ids, partition loss, and cluster lists.

    ``loss`` is the partition energy in **MeV** (same scale as :func:`~cluster_energy.partition_loss_numpy`).
    """

    node_labels: np.ndarray
    loss: float
    partition: list[list[int]]


def make_event_baseline(
    pos: np.ndarray,
    mom: np.ndarray,
    is_proton: np.ndarray,
    partition: list[list[int]],
) -> EventBaseline:
    """Node labels + partition loss from a fixed clustering (``partition`` lists global indices)."""
    n_ev = int(pos.shape[0])
    node_lab = np.empty((n_ev,), dtype=np.int32)
    for ci, clus in enumerate(partition):
        node_lab[np.asarray(clus, dtype=np.int64)] = int(ci)
    loss = float(partition_loss_numpy(pos, mom, is_proton, partition))
    return EventBaseline(node_labels=node_lab, loss=loss, partition=partition)


def edge_pair_baseline_targets(
    node_labels: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
) -> torch.Tensor:
    """Per-edge binary targets: 1 iff baseline puts both endpoints in the same cluster (CPU)."""
    tgt = (node_labels[edge_i] == node_labels[edge_j]).astype(np.float32)
    return torch.tensor(tgt, dtype=torch.float32)
