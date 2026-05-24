import enum
import numpy as np
import torch
import torch.nn.functional as F

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from models.heuristics.utils import edge_pair_baseline_targets
from models.constants import MEV_PER_GEV
from models.env import AffinityGraphEnv
from models.policy import GATAffinityPolicy

class RLActionMode(enum.StrEnum):
    BERNOULLI = enum.auto()
    THRESHOLD = enum.auto()

@dataclass(frozen=True)
class ValueWarmupConfig:
    steps: int = 0
    lr: float = 3e-3
    episodes_per_step: int | None = None

def evaluate_validation_deterministic_policy(
    policy: GATAffinityPolicy,
    env: AffinityGraphEnv,
    events: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, float]:
    policy.eval()
    partition_losses: list[float] = []
    baseline_losses: list[float] = []
    cluster_counts: list[float] = []
    try:
        with torch.no_grad():
            for i, (pos, mom, isp) in enumerate(events):
                obs = env.reset(pos, mom, isp, event_index=int(i))
                logits = policy(obs)
                deterministic_actions = (torch.sigmoid(logits) > 0.5).float()
                partition_loss, cluster_labels = env.physics_for_edge_mask(
                    deterministic_actions
                )
                baseline_loss = float(env._baseline_loss)
                partition_losses.append(float(partition_loss))
                baseline_losses.append(baseline_loss)
                cluster_counts.append(float(len(np.unique(cluster_labels))))
    finally:
        policy.train()
    return {
        "val_n_events": float(len(events)),
        "val_mean_partition_loss_mev": float(np.mean(partition_losses)),
        "val_mean_baseline_loss_mev": float(np.mean(baseline_losses)),
        "val_mean_n_clusters": float(np.mean(cluster_counts)),
    }

def format_validation_console_line(
    *,
    iter_key: str,
    iter_value: int,
    partition_loss_mev: float,
    baseline_loss_mev: float,
    n_events: float | int,
) -> str:
    return (
        f"[val] {iter_key}={iter_value} "
        f"L_pos={partition_loss_mev / MEV_PER_GEV:.4g} "
        f"L_base={baseline_loss_mev / MEV_PER_GEV:.4g} "
        f"(GeV) n={int(n_events)}"
    )

class BestValPhysicsCheckpoint:
    def __init__(
        self,
        path: str | Path,
        *,
        initial_best_loss_mev: float | None = None,
    ) -> None:
        self.path = Path(path)
        self.best_loss_mev: float | None = initial_best_loss_mev

    def maybe_save(
        self,
        val_mean_partition_loss_mev: float,
        state_dict: dict[str, Any],
        *,
        extra: dict[str, Any] | None = None,
    ) -> float | None:
        loss = float(val_mean_partition_loss_mev)
        if not np.isfinite(loss):
            raise ValueError(
                "val_mean_partition_loss_mev must be finite, "
                f"got {val_mean_partition_loss_mev!r}"
            )
        if self.best_loss_mev is not None and loss >= self.best_loss_mev:
            return None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "state_dict": state_dict,
            "val_mean_partition_loss_mev": loss,
        }
        if extra:
            payload.update(extra)
        torch.save(payload, self.path)
        self.best_loss_mev = loss
        return loss

def save_best_checkpoint(
    path: str | Path,
    val_mean_partition_loss_mev: float,
    state_dict: dict[str, Any],
    *,
    best_loss_mev_so_far: float | None,
    extra: dict[str, Any] | None = None,
) -> float | None:
    return BestValPhysicsCheckpoint(
        path, initial_best_loss_mev=best_loss_mev_so_far
    ).maybe_save(val_mean_partition_loss_mev, state_dict, extra=extra)

def eval_physics_loss_deterministic_dataset(
    policy: GATAffinityPolicy,
    env: AffinityGraphEnv,
    events: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> dict[str, float]:
    policy.eval()
    partition_losses: list[float] = []
    for i, (pos, mom, isp) in enumerate(events):
        obs = env.reset(pos, mom, isp, event_index=i)
        logits = policy(obs)
        deterministic_actions = (torch.sigmoid(logits) > 0.5).float()
        partition_loss, _ = env.physics_for_edge_mask(deterministic_actions)
        partition_losses.append(float(partition_loss))
    policy.train()
    arr = np.asarray(partition_losses, dtype=np.float64)
    return {
        "mean_loss_mev": float(arr.mean()),
        "best_loss_mev": float(arr.min()),
        "worst_loss_mev": float(arr.max()),
        "std_loss_mev": float(arr.std(ddof=0)),
        "n_events": int(len(events)),
    }

def baseline_edge_targets(env: AffinityGraphEnv) -> torch.Tensor:
    return edge_pair_baseline_targets(
        env._baseline_node_labels,
        env.graph.edge_pair_i.numpy(),
        env.graph.edge_pair_j.numpy(),
    )

def weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    auto_pos_weight: bool = True,
    pos_weight: float | None = None,
    pos_weight_power: float = 0.5,
    max_pos_weight: float = 300.0,
    focal_gamma: float = 0.0,
) -> tuple[torch.Tensor, float]:
    if pos_weight is not None:
        pw = float(max(1.0, pos_weight))
    elif auto_pos_weight:
        pos = float(targets.sum().item())
        neg = float((1.0 - targets).sum().item())
        ratio = neg / max(pos, 1.0)
        pw = ratio**float(max(pos_weight_power, 0.0))
    else:
        pw = 1.0
    pw = float(np.clip(pw, 1.0, max_pos_weight))
    weights = 1.0 + (pw - 1.0) * targets
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    if focal_gamma > 0.0:
        p = torch.sigmoid(logits)
        pt = torch.where(targets > 0.5, p, 1.0 - p).clamp(1e-6, 1.0 - 1e-6)
        bce = bce * (1.0 - pt) ** float(focal_gamma)
    loss = (bce * weights).sum() / torch.clamp(weights.sum(), min=1.0)
    return loss, pw

type EventSampler = Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray]]
type EventSamplerIndexed = Callable[[], tuple[int, np.ndarray, np.ndarray, np.ndarray]]

def make_event_sampler(
    events: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    rng: np.random.Generator,
    fallback_urqmd: Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> EventSampler:
    def sample_event() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if events:
            pos, mom, isp = events[int(rng.integers(0, len(events)))]
            return pos.copy(), mom.copy(), isp.copy()
        pos, mom, isp = fallback_urqmd()
        return pos, mom, isp

    return sample_event

def make_event_sampler_indexed(
    events: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    rng: np.random.Generator,
) -> EventSamplerIndexed:
    def sample_event() -> tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        if not events:
            raise ValueError("make_event_sampler_indexed requires non-empty events")
        event_index = int(rng.integers(0, len(events)))
        pos, mom, isp = events[event_index]
        return event_index, pos.copy(), mom.copy(), isp.copy()

    return sample_event
