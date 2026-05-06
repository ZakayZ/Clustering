"""Shared helpers for RL and supervised edge training (eval, BCE, sampling, grad stats)."""

import math
import enum
from collections import defaultdict
from typing import Any, Callable, Literal


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.baseline import edge_pair_baseline_targets
from models.env import AffinityGraphEnv
from models.policy import GATAffinityPolicy


class RLActionMode(enum.StrEnum):
    BERNOULLI = enum.auto()
    THRESHOLD = enum.auto()


def deterministic_eval_mean(
    policy: GATAffinityPolicy,
    env: AffinityGraphEnv,
    event_sampler: Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray]],
    n_rollouts: int,
) -> tuple[float, float, float]:
    """Mean return (MeV), mean L_pol (MeV), mean gap (MeV) with deterministic threshold masks."""

    policy.eval()
    rets: list[float] = []
    gaps: list[float] = []
    lp: list[float] = []
    for _ in range(max(int(n_rollouts), 1)):
        pos, mom, isp = event_sampler()
        obs = env.reset(pos, mom, isp)
        logits = policy(obs)
        a = (torch.sigmoid(logits) > 0.5).float()
        loss, _ = env.physics_for_edge_mask(a)
        lb = float(env._baseline_loss)
        Lp = float(loss)
        lp.append(Lp)
        gaps.append(Lp - lb)
        rets.append(float(-Lp))
    policy.train()
    return float(np.mean(rets)), float(np.mean(lp)), float(np.mean(gaps))


def _total_grad_norm(params: list[torch.Tensor]) -> float:
    s = 0.0
    for p in params:
        if p.grad is None:
            continue
        s += float(p.grad.detach().pow(2).sum().item())
    return math.sqrt(s)


def baseline_edge_targets(env: AffinityGraphEnv) -> torch.Tensor:
    """Per-edge binary targets from the spatial–momentum baseline on the current env event.

    Requires ``env.reset`` to have been called for this event so baseline fields are populated.
    """
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
    """BCE over edges with optional positive-class reweighting and focal modulation."""
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


def _tensor_stats_f64(t: torch.Tensor) -> dict[str, float]:
    x = t.detach().float().cpu().reshape(-1)
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "numel": 0.0}
    return {
        "mean": float(x.mean()),
        "std": float(x.std(unbiased=False)),
        "min": float(x.min()),
        "max": float(x.max()),
        "numel": float(x.numel()),
    }


def _policy_grad_norm_by_prefix(policy: nn.Module) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    for name, p in policy.named_parameters():
        if p.grad is None:
            continue
        sums[name.split(".", 1)[0]] += float(p.grad.detach().pow(2).sum().item())
    return {f"grad_norm_{k}": math.sqrt(v) for k, v in sums.items()}


def _summarize_supervised_capture(capture: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in ("node_x", "edge_attr", "h", "le_full", "logits_raw", "logits_out"):
        if k in capture and isinstance(capture[k], torch.Tensor):
            out[k] = _tensor_stats_f64(capture[k])
    for k in ("n_nodes", "n_dir_edges", "n_policy_edges"):
        if k in capture:
            out[k] = capture[k]
    if "logits_raw" in capture and "logits_out" in capture:
        r = capture["logits_raw"].detach().float().reshape(-1)
        o = capture["logits_out"].detach().float().reshape(-1)
        out["frac_logits_changed_postprocess"] = float((~torch.isclose(r, o, rtol=0.0, atol=1e-7)).float().mean())
    return out


type EventSampler = Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray]]


def make_event_sampler(
    events: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    rng: np.random.Generator,
    fallback_urqmd: Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
) -> EventSampler:
    """Build event sampler from preloaded events with optional URQMD fallback."""

    def sample_event() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if events:
            pos, mom, isp = events[int(rng.integers(0, len(events)))]
            return pos.copy(), mom.copy(), isp.copy()
        pos, mom, isp = fallback_urqmd()
        return pos, mom, isp

    return sample_event
