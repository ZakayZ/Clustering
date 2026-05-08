"""Clustering heuristics: cut/coalescence graph, simulated annealing, MST, env baselines.

Import concrete types from submodules (e.g. ``models.heuristics.coalescence.CoalescenceHeuristicModel``)
or use :mod:`models` re-exports. This package ``__init__`` stays minimal to avoid
import cycles.

Graph topologies for :class:`~models.env.AffinityGraphEnv` live in ``models.graph``
(``GraphBuilder`` / ``KNNGraphBuilder`` / ``RadiusGraphBuilder`` / ``FullGraphBuilder``).
"""
