"""Pluggable graph topologies for :class:`~models.env.AffinityGraphEnv`.

Concrete builders correspond to the former ``graph_kind`` switch (kNN / radius / full): each
implements :meth:`GraphBuilder.build` from cut-normalized ``r3`` and ``k3`` (see env ``reset``).
"""

from __future__ import annotations

from models.graph.base import GraphBuilder
from models.graph.full import FullGraphBuilder
from models.graph.knn import KNNGraphBuilder
from models.graph.radius import RadiusGraphBuilder
from models.graph.utils import policy_edges_from_directed

__all__ = [
    "FullGraphBuilder",
    "GraphBuilder",
    "KNNGraphBuilder",
    "RadiusGraphBuilder",
    "policy_edges_from_directed",
]
