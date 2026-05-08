"""BaselineModel protocol: any callable producing :class:`~models.heuristics.utils.EventBaseline`."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from models.heuristics.utils import EventBaseline


class BaselineModel(Protocol):
    """Callable baseline: ``(pos, mom, is_proton, *, event_index=None) -> EventBaseline``.

    Stateless baselines ignore ``event_index``. Indexed caches (e.g.
    :class:`~models.heuristics.static_partition.StaticPartitionBaseline`) use it when set.
    """

    def __call__(
        self,
        pos: np.ndarray,
        mom: np.ndarray,
        is_proton: np.ndarray,
        *,
        event_index: int | None = None,
    ) -> EventBaseline: ...
