"""Abstract graph builder for :class:`~models.env.AffinityGraphEnv`."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from torch_geometric.data import Data


class GraphBuilder(ABC):
    """Builds PyG ``Data`` with ``edge_index`` and policy edge bookkeeping (undirected pairs)."""

    @abstractmethod
    def build(self, r3: np.ndarray, k3: np.ndarray) -> Data:
        """``r3``/``k3`` are cut-normalized (see :meth:`models.env.AffinityGraphEnv.reset`)."""
        raise NotImplementedError()
