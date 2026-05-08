"""Supervised edge-BCE training against the spatial–momentum baseline (warm-start / standalone).

Per-step ``pretrain_partition_loss`` uses :meth:`~models.env.AffinityGraphEnv.physics_for_edge_mask`,
so optional ``cluster_dissolve_energy_threshold_mev`` on the env affects logged physics loss.
"""

import numpy as np
import torch
import torch.nn as nn

from typing import Callable

from torch.optim.lr_scheduler import LRScheduler
from tqdm.auto import tqdm

from models import AffinityGraphEnv, GATAffinityPolicy, MEV_PER_GEV

from .utils import (
    EventSampler,
    baseline_edge_targets,
    evaluate_validation_deterministic_policy,
    weighted_bce_with_logits,
)


def train_supervised_edges(
    policy: GATAffinityPolicy,
    env: AffinityGraphEnv,
    event_sampler: EventSampler,
    *,
    steps: int,
    events_per_step: int = 8,
    lr: float = 3e-3,
    max_grad_norm: float = 0.5,
    weighted_bce: bool = True,
    pos_weight: float | None = None,
    pos_weight_power: float = 0.5,
    pos_weight_max: float = 300.0,
    focal_gamma: float = 0.0,
    on_update: Callable[[dict[str, list]], None] | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    lr_scheduler: LRScheduler | None = None,
    val_events: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    n_val_steps: int = 1,
) -> dict[str, list]:
    """BCE on edge logits against spatial–momentum baseline edge targets (warm-start / standalone).

    If ``lr_scheduler`` is set, ``lr_scheduler.step()`` runs after each ``optimizer.step()``.

    Validation: non-empty ``val_events`` and ``n_val_steps > 0`` runs
    :func:`training.utils.evaluate_validation_deterministic_policy` every ``n_val_steps`` training
    steps (slice ``val_events`` yourself). Only deterministic-mask physics metrics are recorded.
    Use ``n_val_steps <= 0`` to disable.
    """
    policy_optimizer = optimizer or torch.optim.Adam(policy.parameters(), lr=lr)
    history: dict[str, list] = {
        "supervised_bce": [],
        "pretrain_partition_loss": [],
        "pretrain_baseline_loss": [],
        "supervised_pos_weight": [],
        "pretrain_pos_recall_05": [],
        "pretrain_mean_prob_pos_edges": [],
        "pretrain_mean_prob_neg_edges": [],
        "pretrain_frac_pred_on_05": [],
        "pretrain_mean_n_clusters": [],
    }
    val_cfg = (
        val_events is not None
        and len(val_events) > 0
        and int(n_val_steps) > 0
    )
    if val_cfg:
        history["val_step"] = []
        history["val_mean_partition_loss_mev"] = []
        history["val_mean_baseline_loss_mev"] = []
        history["val_mean_n_clusters"] = []
        history["val_n_events"] = []
    policy.train()
    pbar_sup = tqdm(
        range(steps),
        desc="SupEdges",
        miniters=1,
        mininterval=0.0,
        dynamic_ncols=True,
    )
    for step_ix in pbar_sup:
        policy_optimizer.zero_grad()
        supervised_loss_sum = 0.0
        partition_losses: list[float] = []
        baseline_losses: list[float] = []
        num_events_in_step = 0
        positive_recall_sum = 0.0
        num_positive_recall_batches = 0
        mean_prob_on_positive_edges_sum = 0.0
        num_positive_mean_prob_batches = 0
        mean_prob_on_negative_edges_sum = 0.0
        num_negative_mean_prob_batches = 0
        frac_predicted_on_sum = 0.0
        cluster_count_sum = 0.0
        for _microbatch in range(events_per_step):
            raw = event_sampler()
            if len(raw) == 4:
                ev_idx, pos, mom, isp = raw
                obs = env.reset(pos, mom, isp, event_index=int(ev_idx))
            else:
                pos, mom, isp = raw  # type: ignore[misc]
                obs = env.reset(pos, mom, isp)
            logits = policy(obs)
            edge_on = (torch.sigmoid(logits) > 0.5).float()
            partition_loss, cluster_labels = env.physics_for_edge_mask(edge_on)
            partition_losses.append(float(partition_loss))
            baseline_losses.append(float(env._baseline_loss))
            tgt = baseline_edge_targets(env)
            loss_sup, pw = weighted_bce_with_logits(
                logits,
                tgt,
                auto_pos_weight=weighted_bce and pos_weight is None,
                pos_weight=pos_weight,
                pos_weight_power=pos_weight_power,
                max_pos_weight=pos_weight_max,
                focal_gamma=focal_gamma,
            )
            loss_sup.backward()
            supervised_loss_sum += float(loss_sup.item())
            history["supervised_pos_weight"].append(float(pw))
            num_events_in_step += 1
            with torch.no_grad():
                prob = torch.sigmoid(logits)
                positive_edges = tgt > 0.5
                negative_edges = ~positive_edges
                num_positive_edges = int(positive_edges.sum().item())
                if num_positive_edges > 0:
                    true_positives = ((prob > 0.5) & positive_edges).sum().item()
                    positive_recall_sum += float(true_positives) / float(num_positive_edges)
                    num_positive_recall_batches += 1
                    mean_prob_on_positive_edges_sum += float(prob[positive_edges].mean().item())
                    num_positive_mean_prob_batches += 1
                num_negative_edges = int(negative_edges.sum().item())
                if num_negative_edges > 0:
                    mean_prob_on_negative_edges_sum += float(
                        prob[negative_edges].mean().item()
                    )
                    num_negative_mean_prob_batches += 1
                frac_predicted_on_sum += float((prob > 0.5).float().mean().item())
                cluster_count_sum += float(len(np.unique(cluster_labels)))
        nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        policy_optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        mean_supervised_loss = supervised_loss_sum / max(float(num_events_in_step), 1.0)
        history["supervised_bce"].append(mean_supervised_loss)
        if partition_losses:
            history["pretrain_partition_loss"].append(float(np.mean(partition_losses)))
        if baseline_losses:
            history["pretrain_baseline_loss"].append(float(np.mean(baseline_losses)))
        normalization_events = max(float(num_events_in_step), 1.0)
        history["pretrain_pos_recall_05"].append(
            positive_recall_sum / max(float(num_positive_recall_batches), 1.0)
        )
        history["pretrain_mean_prob_pos_edges"].append(
            mean_prob_on_positive_edges_sum / max(float(num_positive_mean_prob_batches), 1.0)
        )
        history["pretrain_mean_prob_neg_edges"].append(
            mean_prob_on_negative_edges_sum / max(float(num_negative_mean_prob_batches), 1.0)
        )
        history["pretrain_frac_pred_on_05"].append(frac_predicted_on_sum / normalization_events)
        history["pretrain_mean_n_clusters"].append(cluster_count_sum / normalization_events)
        progress_metrics: dict[str, float] = {}
        if partition_losses:
            progress_metrics["partition_gev"] = (
                float(np.mean(partition_losses)) / MEV_PER_GEV
            )
        if baseline_losses:
            progress_metrics["baseline_gev"] = (
                float(np.mean(baseline_losses)) / MEV_PER_GEV
            )
        progress_metrics["bce"] = mean_supervised_loss
        if history["pretrain_pos_recall_05"]:
            progress_metrics["positive_recall"] = float(history["pretrain_pos_recall_05"][-1])
        if history["pretrain_frac_pred_on_05"]:
            progress_metrics["frac_edges_pred_on"] = float(history["pretrain_frac_pred_on_05"][-1])
        pbar_sup.set_postfix(progress_metrics, refresh=True)
        if val_cfg and step_ix % int(n_val_steps) == 0:
            validation_metrics = evaluate_validation_deterministic_policy(policy, env, val_events)
            if validation_metrics:
                history["val_step"].append(int(step_ix))
                history["val_mean_partition_loss_mev"].append(
                    validation_metrics["val_mean_partition_loss_mev"]
                )
                history["val_mean_baseline_loss_mev"].append(
                    validation_metrics["val_mean_baseline_loss_mev"]
                )
                history["val_mean_n_clusters"].append(
                    validation_metrics["val_mean_n_clusters"]
                )
                history["val_n_events"].append(validation_metrics["val_n_events"])
                tqdm.write(
                    f"[val] step={step_ix} L_pos={validation_metrics['val_mean_partition_loss_mev']:.4g} "
                    f"L_base={validation_metrics['val_mean_baseline_loss_mev']:.4g} "
                    f"n={int(validation_metrics['val_n_events'])}"
                )
        if on_update is not None:
            on_update(history)
    return history
