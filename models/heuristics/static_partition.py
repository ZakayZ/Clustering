import gzip
import pickle

from pathlib import Path
from typing import Any, Self

import numpy as np

from models.heuristics.utils import EventBaseline, make_event_baseline

class StaticPartitionBaseline:
    def __init__(self, cache: list[EventBaseline]) -> None:
        self._cache = cache
        self._current_index: int = 0

    def _set_current_index(self, index: int) -> None:
        idx = int(index)
        if idx < 0 or idx >= len(self._cache):
            raise IndexError(f"event index {idx} out of range [0, {len(self._cache)})")
        self._current_index = idx

    def prepare_event(self, index: int) -> None:
        self._set_current_index(index)

    def __call__(
        self,
        pos: np.ndarray,
        mom: np.ndarray,
        is_proton: np.ndarray,
        *,
        event_index: int | None = None,
    ) -> EventBaseline:
        if event_index is not None:
            self._set_current_index(event_index)
        return self._cache[self._current_index]

    @classmethod
    def from_heuristic_bundle(
        cls,
        teacher_pkl_gz: Path | str,
        events: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> Self:
        path = Path(teacher_pkl_gz)
        with gzip.open(path, "rb") as f:
            bundle: dict[str, Any] = pickle.load(f)
        rows: list[dict[str, Any]] = list(bundle["events"])
        rows.sort(key=lambda r: int(r["event_idx"]))
        if len(rows) != len(events):
            raise ValueError(
                f"teacher has {len(rows)} events, dataset has {len(events)} — paths must match"
            )
        cache: list[EventBaseline] = []
        for i, row in enumerate(rows):
            if int(row["event_idx"]) != i:
                raise ValueError(f"expected event_idx {i}, got {row['event_idx']}")
            pos, mom, isp = events[i]
            pos = np.asarray(pos, dtype=np.float64)
            mom = np.asarray(mom, dtype=np.float64)
            isp = np.asarray(isp, dtype=bool)
            part = row["partition"]
            cache.append(make_event_baseline(pos, mom, isp, part))
        return cls(cache)
