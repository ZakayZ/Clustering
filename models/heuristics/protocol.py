"""BaselineModel protocol: any callable producing :class:`~models.heuristics.utils.EventBaseline`."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from models.heuristics.utils import EventBaseline


class BaselineModel(Protocol):
    """Callable baseline: ``(pos, mom, is_proton) -> EventBaseline``."""

    def __call__(
        self,
        pos: np.ndarray,
        mom: np.ndarray,
        is_proton: np.ndarray,
    ) -> EventBaseline: ...
