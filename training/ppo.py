"""Proximal Policy Optimization (PPO) for affinity graph edge policies."""

import numpy as np
import torch
import torch.nn as nn

from typing import Any, Callable

from torch.optim.lr_scheduler import LRScheduler
from tqdm.auto import tqdm

from models import AffinityGraphEnv, GATAffinityActorCritic, MEV_PER_GEV

from .a2c import warm_start_value_head
from .utils import RLActionMode, ValueWarmupConfig, evaluate_validation_deterministic_policy

EventSampler = Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray]]


def collect_rollout_ppo(
    env: AffinityGraphEnv,
    ac: GATAffinityActorCritic,
    pos: np.ndarray,
    mom: np.ndarray,
    isp: np.ndarray,
    *,
    action_mode: RLActionMode = "bernoulli",
) -> dict[str, Any]:
    """PPO rollout: ``value``, ``old_log_prob`` (mean edge log π_i(a_i)), plus diagnostics."""
    obs = env.reset(pos, mom, isp)
    edge_logits, state_value = ac(obs)
    dist = torch.distributions.Bernoulli(logits=edge_logits)
    if action_mode == "threshold":
        edge_actions = (torch.sigmoid(edge_logits) > 0.5).float()
    else:
        edge_actions = dist.sample()
    mean_edge_entropy = dist.entropy().mean()
    old_log_prob = dist.log_prob(edge_actions).mean().detach()
    partition_loss, cluster_labels = env.physics_for_edge_mask(edge_actions)
    baseline_loss = float(env._baseline_loss)
    reward = -partition_loss
    return {
        "obs": obs,
        "action": edge_actions,
        "reward": reward,
        "value": float(state_value.detach().item()),
        "old_log_prob": old_log_prob,
        "partition_loss": partition_loss,
        "baseline_loss": baseline_loss,
        "edge_entropy": mean_edge_entropy,
        "n_clusters": int(len(np.unique(cluster_labels))),
        "action_mode": action_mode,
        "event_pos": np.asarray(pos, dtype=np.float64, copy=True),
        "event_mom": np.asarray(mom, dtype=np.float64, copy=True),
        "event_isp": np.asarray(isp, copy=True),
    }


def train_ppo(
    ac: GATAffinityActorCritic,
    env: AffinityGraphEnv,
    event_sampler: EventSampler,
    *,
    optimizer: torch.optim.Optimizer,
    n_updates: int = 150,
    episodes_per_update: int = 8,
    ppo_epochs: int = 1,
    clip_range: float = 0.2,
    value_coef: float = 0.5,
    ent_coef: float = 0.02,
    policy_coef: float = 1.0,
    max_grad_norm: float = 0.5,
    normalize_advantage: bool = True,
    value_warmup: ValueWarmupConfig | None = None,
    detach_value_features: bool = True,
    log_ratio_clip: float = 20.0,
    lr_scheduler: LRScheduler | None = None,
    on_update: Callable[[dict[str, list]], None] | None = None,
    rl_action_mode: RLActionMode = "bernoulli",
    val_events: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    n_val_steps: int = 1,
) -> dict[str, list]:
    """Proximal Policy Optimization (Schulman et al.) on edge Bernoulli masks.

    Each environment step is one full graph: sample all edge bits, observe terminal reward
    ``R = -\\mathcal{L}_\\mathrm{pol}`` (MeV). Advantages are ``A = R - V(s)`` from rollout values
    (Monte Carlo / one-step TD). Policy log-probs are **mean** over edges; the importance ratio
    uses ``r = \\exp(N(\\overline{\\log\\pi_\\theta} - \\overline{\\log\\pi_{\\mathrm{old}}}))``
    with ``N`` = number of policy edges (factorized Bernoulli). Each PPO epoch shuffles the rollout
    buffer and applies one gradient step on the **full** batch (all ``episodes_per_update`` graphs).

    ``lr_scheduler.step()`` runs **once per outer update** after PPO optimization.

    Optional ``value_warmup`` with ``steps > 0`` runs :func:`training.a2c.warm_start_value_head`
    once at the start so the critic matches MeV-scale returns before PPO updates. Pass ``None``
    (default) to skip.

    Validation: non-empty ``val_events`` and ``n_val_steps > 0`` evaluates ``ac.policy`` every
    ``n_val_steps`` outer updates (slice ``val_events`` yourself). Use ``n_val_steps <= 0`` to disable.
    """
    value_warmup_cfg = value_warmup or ValueWarmupConfig()
    if int(value_warmup_cfg.steps) > 0:
        warmup_episodes_per_step = (
            int(value_warmup_cfg.episodes_per_step)
            if value_warmup_cfg.episodes_per_step is not None
            else int(episodes_per_update)
        )
        warm_start_value_head(
            ac,
            env,
            event_sampler,
            steps=int(value_warmup_cfg.steps),
            episodes_per_step=max(1, warmup_episodes_per_step),
            lr=float(value_warmup_cfg.lr),
            max_grad_norm=float(max_grad_norm),
            rl_action_mode=rl_action_mode,
        )

    history: dict[str, list] = {
        "episode_return": [],
        "partition_loss": [],
        "baseline_loss": [],
        "policy_loss": [],
        "value_loss": [],
        "value_mean": [],
        "advantage_mean": [],
        "edge_entropy": [],
        "approx_kl": [],
        "clip_fraction": [],
        "n_clusters": [],
        "lr": [],
    }
    val_cfg = (
        val_events is not None
        and len(val_events) > 0
        and int(n_val_steps) > 0
    )
    if val_cfg:
        history["val_step"] = []
        for key in (
            "val_n_events",
            "val_mean_partition_loss_mev",
            "val_mean_baseline_loss_mev",
            "val_mean_n_clusters",
        ):
            history[key] = []
    pbar = tqdm(range(n_updates), desc="PPO", miniters=1, mininterval=0.0, dynamic_ncols=True)
    for u in pbar:
        buffer: list[dict[str, Any]] = []
        episode_returns: list[float] = []
        partition_losses: list[float] = []
        baseline_losses: list[float] = []
        cluster_counts: list[int] = []
        rollout_values: list[float] = []

        for _ in range(episodes_per_update):
            pos, mom, isp = event_sampler()
            with torch.no_grad():
                ep = collect_rollout_ppo(
                    env,
                    ac,
                    pos,
                    mom,
                    isp,
                    action_mode=rl_action_mode,
                )
            buffer.append(ep)
            episode_returns.append(float(ep["reward"]))
            partition_losses.append(float(ep["partition_loss"]))
            baseline_losses.append(float(ep["baseline_loss"]))
            cluster_counts.append(ep["n_clusters"])
            rollout_values.append(float(ep["value"]))

        num_rollouts = len(buffer)
        episode_returns_np = np.asarray(
            [float(ep["reward"]) for ep in buffer], dtype=np.float64
        )
        rollout_values_np = np.asarray([float(ep["value"]) for ep in buffer], dtype=np.float64)
        advantages_np = episode_returns_np - rollout_values_np
        if normalize_advantage and num_rollouts > 1:
            advantages_np = (advantages_np - advantages_np.mean()) / (
                float(advantages_np.std()) + 1e-8
            )

        ac.train()
        dtype = next(ac.parameters()).dtype

        policy_loss_sum = 0.0
        value_loss_sum = 0.0
        entropy_sum = 0.0
        approx_kl_sum = 0.0
        clip_fraction_sum = 0.0
        num_surrogate_terms = 0

        for _ in range(ppo_epochs):
            shuffled_indices = np.random.permutation(num_rollouts)
            optimizer.zero_grad()
            accumulated_loss = torch.zeros((), dtype=dtype)
            normalization_divisor = max(num_rollouts, 1)
            for rollout_idx in shuffled_indices:
                ep = buffer[int(rollout_idx)]
                obs = ep["obs"]
                actions = ep["action"]
                old_log_prob_tensor = ep["old_log_prob"].to(dtype=dtype)
                episode_return = float(ep["reward"])
                advantage_scalar = float(advantages_np[int(rollout_idx)])

                logits, values = ac(obs, detach_value_features=bool(detach_value_features))
                dist = torch.distributions.Bernoulli(logits=logits)
                mean_log_prob = dist.log_prob(actions).mean()
                mean_entropy = dist.entropy().mean()
                num_edges = int(actions.numel())

                advantage = torch.tensor(advantage_scalar, dtype=dtype)
                return_tensor = torch.tensor(episode_return, dtype=dtype)

                log_ratio = num_edges * (mean_log_prob - old_log_prob_tensor)
                if float(log_ratio_clip) > 0.0:
                    log_ratio = torch.clamp(
                        log_ratio,
                        -float(log_ratio_clip),
                        float(log_ratio_clip),
                    )
                ratio = torch.exp(log_ratio)
                surrogate_1 = ratio * advantage
                surrogate_2 = (
                    torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantage
                )
                clipped_surrogate_loss = -torch.min(surrogate_1, surrogate_2)

                value_loss = 0.5 * (values - return_tensor).pow(2)

                step_loss = (
                    policy_coef * clipped_surrogate_loss
                    + value_coef * value_loss
                    - ent_coef * mean_entropy
                ) / normalization_divisor
                accumulated_loss = accumulated_loss + step_loss

                with torch.no_grad():
                    policy_loss_sum += float(clipped_surrogate_loss.item())
                    value_loss_sum += float(value_loss.item())
                    entropy_sum += float(mean_entropy.item())
                    approx_kl_sum += float((num_edges * (old_log_prob_tensor - mean_log_prob)).item())
                    ratio_far_from_one = torch.abs(ratio - 1.0) > clip_range
                    clip_fraction_sum += float(ratio_far_from_one.float().mean().item())
                    num_surrogate_terms += 1

            accumulated_loss.backward()
            nn.utils.clip_grad_norm_(ac.policy.parameters(), max_grad_norm)
            nn.utils.clip_grad_norm_(ac.value_head.parameters(), max_grad_norm)
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if val_cfg and u % int(n_val_steps) == 0:
            validation_metrics = evaluate_validation_deterministic_policy(
                ac.policy, env, val_events
            )
            if validation_metrics:
                history["val_step"].append(int(u))
                for key in (
                    "val_n_events",
                    "val_mean_partition_loss_mev",
                    "val_mean_baseline_loss_mev",
                    "val_mean_n_clusters",
                ):
                    history[key].append(validation_metrics[key])
                tqdm.write(
                    f"[val] update={u} L_pos={validation_metrics['val_mean_partition_loss_mev']:.4g} "
                    f"L_base={validation_metrics['val_mean_baseline_loss_mev']:.4g} "
                    f"n={int(validation_metrics['val_n_events'])}"
                )

        surrogate_denominator = max(num_surrogate_terms, 1)
        if episode_returns:
            history["episode_return"].append(float(np.mean(episode_returns)))
        if partition_losses:
            history["partition_loss"].append(float(np.mean(partition_losses)))
        if baseline_losses and all(np.isfinite(baseline_losses)):
            history["baseline_loss"].append(float(np.mean(baseline_losses)))
        if cluster_counts:
            history["n_clusters"].append(float(np.mean(cluster_counts)))
        history["policy_loss"].append(policy_loss_sum / surrogate_denominator)
        history["value_loss"].append(value_loss_sum / surrogate_denominator)
        history["value_mean"].append(
            float(np.mean(rollout_values)) if rollout_values else 0.0
        )
        history["advantage_mean"].append(
            float(np.mean(advantages_np)) if num_rollouts else 0.0
        )
        history["edge_entropy"].append(entropy_sum / surrogate_denominator)
        history["approx_kl"].append(approx_kl_sum / surrogate_denominator)
        history["clip_fraction"].append(clip_fraction_sum / surrogate_denominator)

        progress_metrics: dict[str, float] = {}
        if partition_losses:
            progress_metrics["partition_gev"] = (
                float(np.mean(partition_losses)) / MEV_PER_GEV
            )
        if baseline_losses and all(np.isfinite(baseline_losses)):
            progress_metrics["baseline_gev"] = (
                float(np.mean(baseline_losses)) / MEV_PER_GEV
            )
        progress_metrics["policy_loss"] = policy_loss_sum / surrogate_denominator
        progress_metrics["value_loss"] = value_loss_sum / surrogate_denominator
        progress_metrics["approx_kl"] = approx_kl_sum / surrogate_denominator
        progress_metrics["clip_frac"] = clip_fraction_sum / surrogate_denominator
        progress_metrics["entropy"] = entropy_sum / surrogate_denominator
        progress_metrics["lr"] = float(optimizer.param_groups[0]["lr"])
        if rollout_values:
            progress_metrics["mean_V_gev"] = float(np.mean(rollout_values)) / MEV_PER_GEV
        if episode_returns:
            progress_metrics["mean_return_gev"] = (
                float(np.mean(episode_returns)) / MEV_PER_GEV
            )
        if cluster_counts:
            progress_metrics["mean_clusters"] = float(np.mean(cluster_counts))
        pbar.set_postfix(progress_metrics, refresh=True)
        if on_update is not None:
            on_update(history)

    return history
