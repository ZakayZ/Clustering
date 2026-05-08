"""Simulated annealing on nucleon partitions (:func:`~cluster_energy.partition_loss_numpy`).

CCL-style moves and schedules (``vkireyeu/ccl`` ``makeSA`` / ``makeSA2`` in ``src/MSTUtils.cxx``):

- **Pass 1** (``growth_only=False``): with probability ``p_new`` (scheduled like CCL),
  move a nucleon out of a multi-body cluster into a **new** singleton cluster; otherwise
  transfer one nucleon between distinct clusters.
- **Pass 2** (``growth_only=True``): transfer moves only; greedy acceptance (CCL ``makeSA2``).

Also exports :func:`normalize_partition` / :func:`check_partition` helpers used across baselines.

Does **not** implement CCL ``makeSAchain`` branching on macroscopic binding per cluster.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np

from cluster_energy import partition_loss_numpy

from models.heuristics.utils import EventBaseline, make_event_baseline

if TYPE_CHECKING:
    from models.heuristics.protocol import BaselineModel


def normalize_partition(partition: list[list[int]]) -> list[list[int]]:
    cleaned: list[list[int]] = []
    for c in partition:
        if len(c) == 0:
            continue
        cleaned.append(sorted(set(int(x) for x in c)))
    cleaned.sort(key=lambda x: (len(x), x), reverse=True)
    return cleaned


def check_partition(partition: list[list[int]], universe: list[int]) -> None:
    flat: list[int] = []
    for c in partition:
        flat.extend(int(x) for x in c)
    if sorted(flat) != sorted(int(u) for u in universe):
        raise ValueError("partition does not cover universe")


class SAStepsMode(str, Enum):
    """Analog of ``CClusterizer::SAStepsMode``."""

    FIXED = "fixed"
    LINEAR = "linear"
    EXPONENTIAL = "exponential"


class SAProbMode(str, Enum):
    """Analog of ``CClusterizer::SAProbMode``."""

    FIXED = "fixed"
    LINEAR = "linear"


@dataclass(frozen=True)
class CCLAnnealParams:
    """Hyperparameters mapped from CCL ``libccl.h`` simulated-annealing fields.

    **Defaults** tune post–cut/coalescence refinement with a bounded wall clock per event
    (``max_seconds``); see ``clustering/benchmarks/coalesce_ccl_anneal_search.json``.

    For the older exponential ``CCL`` schedule used as a search baseline,
    see :data:`REFERENCE_CCL_ANNEAL_PARAMS`.
    """

    t_max: float = 1.0
    t_min: float = 5e-7
    cool: float = 0.92
    steps: int = 300
    steps_mode: SAStepsMode = SAStepsMode.FIXED
    p_new: float = 0.38
    p_new_min: float = 0.10
    p_new_mode: SAProbMode = SAProbMode.LINEAR
    use_metropolis_hastings: bool = False
    stag_min: int = 60
    stag_denom: int = 10
    #: Second-pass ``steps``; ``< 0`` means ``2 * steps`` (CCL ``sa_Steps2`` default).
    steps_2: int = 360
    two_phase: bool = True
    rng_seed: int = 0
    max_seconds: float | None = 2.5

    def resolve_steps_2(self) -> int:
        return int(self.steps * 2) if self.steps_2 < 0 else int(self.steps_2)


#: Pre–tuning exponential schedule (``.95`` cooling, exponential steps scaling), **uncapped**.
#: Hyperparameter searches compare randomized trials against this reference + ``max_seconds``.
REFERENCE_CCL_ANNEAL_PARAMS = CCLAnnealParams(
    t_max=1.0,
    t_min=1e-5,
    cool=0.95,
    steps=500,
    steps_mode=SAStepsMode.EXPONENTIAL,
    p_new=0.25,
    p_new_min=0.25,
    p_new_mode=SAProbMode.FIXED,
    use_metropolis_hastings=False,
    stag_min=10,
    stag_denom=10,
    steps_2=-1,
    two_phase=False,
    rng_seed=0,
    max_seconds=None,
)


def _partition_from_map(clusters: dict[int, list[int]]) -> list[list[int]]:
    return normalize_partition([sorted(lst) for lst in clusters.values() if lst])


def _loss(
    pos: np.ndarray,
    mom: np.ndarray,
    is_proton: np.ndarray,
    clusters: dict[int, list[int]],
) -> float:
    return float(partition_loss_numpy(pos, mom, is_proton, _partition_from_map(clusters)))


def _steps_at_level(*, cfg: CCLAnnealParams, sa_level: int, t: float) -> int:
    if cfg.steps_mode == SAStepsMode.FIXED:
        s = cfg.steps
    elif cfg.steps_mode == SAStepsMode.LINEAR:
        s = int(cfg.steps * (t / cfg.t_max))
    else:
        s = int(cfg.steps * (cfg.cool**sa_level))
    return max(1, s)


def _p_new_effective(cfg: CCLAnnealParams, t: float) -> float:
    if cfg.p_new_mode == SAProbMode.FIXED:
        return float(cfg.p_new)
    span = cfg.t_max - cfg.t_min
    if span <= 0.0:
        return float(cfg.p_new_min)
    ratio = (t - cfg.t_min) / span
    return float(cfg.p_new_min + (cfg.p_new - cfg.p_new_min) * ratio)


def _try_propose_transfer(
    clusters: dict[int, list[int]],
    rng: random.Random,
) -> dict[int, list[int]] | None:
    if len(clusters) < 1:
        return None
    active = sorted(clusters.keys())
    cid_src = rng.choice(active)
    body_src = clusters[cid_src]
    idx_pick = rng.randrange(len(body_src)) if len(body_src) > 1 else 0

    cid_dst = rng.choice(active)
    guard = 0
    while cid_dst == cid_src and len(active) > 1 and guard < len(active) * 4:
        cid_dst = rng.choice(active)
        guard += 1
    if cid_dst == cid_src and len(active) > 1:
        alternatives = [c for c in active if c != cid_src]
        cid_dst = rng.choice(alternatives)

    particle = body_src[idx_pick]
    nxt = {k: v[:] for k, v in clusters.items()}
    moved_from = nxt[cid_src]
    moved_from.pop(idx_pick)
    if not moved_from:
        del nxt[cid_src]
    nxt.setdefault(cid_dst, []).append(int(particle))
    nxt[cid_dst] = sorted(set(nxt[cid_dst]))
    return nxt


def _try_propose_split_new(
    clusters: dict[int, list[int]],
    max_id: int,
    rng: random.Random,
) -> tuple[dict[int, list[int]], int] | None:
    multi = [cid for cid, lst in clusters.items() if len(lst) >= 2]
    if not multi:
        return None
    cid_src = rng.choice(multi)
    body = clusters[cid_src]
    if len(body) < 2:
        return None
    ji = rng.randrange(len(body))
    particle = int(body[ji])
    new_id = max_id + 1
    nxt = {k: v[:] for k, v in clusters.items()}
    nxt[cid_src].pop(ji)
    nxt[cid_src].sort()
    if not nxt[cid_src]:
        del nxt[cid_src]
    nxt[new_id] = [particle]
    return nxt, new_id


def ccl_sa_refine_partition(
    pos: np.ndarray,
    mom: np.ndarray,
    is_proton: np.ndarray,
    initial_partition: list[list[int]],
    universe: list[int],
    *,
    cfg: CCLAnnealParams,
    rng_seed: int,
    growth_only: bool = False,
    steps_override: int | None = None,
) -> tuple[list[list[int]], float]:
    """Run one CCL-like SA trajectory; return **best** partition and its loss."""

    rng = random.Random(int(rng_seed) & 0xFFFFFFFF)

    clusters: dict[int, list[int]] = {}
    cur_id = 0
    norm0 = normalize_partition(initial_partition)
    check_partition(norm0, universe)
    for grp in norm0:
        clusters[cur_id] = sorted(int(x) for x in grp)
        cur_id += 1

    n_particles = len(universe)
    max_cluster_id = max(clusters.keys(), default=-1)

    curr_loss = _loss(pos, mom, is_proton, clusters)
    best_clusters = {k: v[:] for k, v in clusters.items()}
    best_loss = curr_loss

    t_wall0 = time.time()
    sa_level = 0
    t = float(cfg.t_max)
    while t > float(cfg.t_min):
        if steps_override is not None:
            if cfg.steps_mode == SAStepsMode.FIXED:
                steps_local = max(1, int(steps_override))
            elif cfg.steps_mode == SAStepsMode.LINEAR:
                steps_local = max(1, int(steps_override * (t / cfg.t_max)))
            else:
                steps_local = max(1, int(steps_override * (cfg.cool**sa_level)))
        else:
            steps_local = _steps_at_level(cfg=cfg, sa_level=sa_level, t=t)

        stagnant = 0
        stag_limit = max(int(cfg.stag_min), steps_local // max(1, int(cfg.stag_denom)))

        p_new_eff = float(_p_new_effective(cfg, t)) if not growth_only else 0.0

        active_count = len(clusters)

        for _ in range(steps_local):
            if cfg.max_seconds is not None and (time.time() - t_wall0) > cfg.max_seconds:
                return normalize_partition([sorted(v) for v in best_clusters.values() if v]), best_loss

            split_branch = False
            if not growth_only:
                u_draw = rng.random()
                if active_count == n_particles:
                    u_draw = p_new_eff * 1.1
                elif active_count == 1:
                    u_draw = p_new_eff * 0.1
                split_branch = u_draw < p_new_eff

            proposed: dict[int, list[int]] | None = None
            new_max_id = max_cluster_id

            if split_branch:
                spl = _try_propose_split_new(clusters, max_cluster_id, rng)
                if spl is None:
                    proposed = _try_propose_transfer(clusters, rng)
                else:
                    proposed, new_max_id = spl
            else:
                proposed = _try_propose_transfer(clusters, rng)

            if proposed is None:
                stagnant += 1
                if stagnant >= stag_limit:
                    break
                continue

            new_loss = _loss(pos, mom, is_proton, proposed)
            delta = new_loss - curr_loss

            accept = delta < 0.0
            if not accept and (not growth_only) and cfg.use_metropolis_hastings:
                if rng.random() < math.exp(-delta / max(t, 1e-300)):
                    accept = True

            if accept:
                clusters = proposed
                max_cluster_id = max(max_cluster_id, new_max_id)
                curr_loss = new_loss
                active_count = len(clusters)
                stagnant = 0
                if curr_loss < best_loss:
                    best_loss = curr_loss
                    best_clusters = {k: v[:] for k, v in clusters.items()}
            else:
                stagnant += 1
                if stagnant >= stag_limit:
                    break

        t *= float(cfg.cool)
        sa_level += 1

    return normalize_partition([sorted(v) for v in best_clusters.values() if v]), best_loss


@dataclass
class CCLAnnealRefinementModel:
    """Wrap any :class:`~models.heuristics.protocol.BaselineModel` with CCL-style SA."""

    inner: BaselineModel
    params: CCLAnnealParams = field(default_factory=CCLAnnealParams)
    _call_idx: int = field(default=0, repr=False)

    def __call__(
        self,
        pos: np.ndarray,
        mom: np.ndarray,
        is_proton: np.ndarray,
        *,
        event_index: int | None = None,
    ) -> EventBaseline:
        n_ev = int(pos.shape[0])
        uni = list(range(n_ev))
        base = self.inner(pos, mom, is_proton, event_index=event_index)
        init = normalize_partition(base.partition)
        check_partition(init, uni)

        seed = int(self.params.rng_seed + self._call_idx)
        self._call_idx += 1

        part, _ = ccl_sa_refine_partition(
            pos,
            mom,
            is_proton,
            init,
            uni,
            cfg=self.params,
            rng_seed=seed,
            growth_only=False,
            steps_override=None,
        )
        if self.params.two_phase:
            seed2 = int(self.params.rng_seed + self._call_idx)
            self._call_idx += 1
            part, _ = ccl_sa_refine_partition(
                pos,
                mom,
                is_proton,
                part,
                uni,
                cfg=self.params,
                rng_seed=seed2,
                growth_only=True,
                steps_override=self.params.resolve_steps_2(),
            )
        return make_event_baseline(pos, mom, is_proton, part)
