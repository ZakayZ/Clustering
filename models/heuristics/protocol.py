from typing import Protocol

import numpy as np

from models.heuristics.utils import EventBaseline

class BaselineModel(Protocol):
    def __call__(
        self,
        pos: np.ndarray,
        mom: np.ndarray,
        is_proton: np.ndarray,
        *,
        event_index: int | None = None,
    ) -> EventBaseline: ...
