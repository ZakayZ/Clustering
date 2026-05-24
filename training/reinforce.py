import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from typing import Any, Callable

from torch.optim.lr_scheduler import LRScheduler
from torch_geometric.data import Data
from tqdm.auto import tqdm

from models import AffinityGraphEnv, GATAffinityPolicy, MEV_PER_GEV

from .utils import (
    RLActionMode,
    BestValPhysicsCheckpoint,
    EventSampler,
    evaluate_validation_deterministic_policy,
    format_validation_console_line,
)

def collect_rollout(
    env: AffinityGraphEnv,
    policy: GATAffinityPolicy,
    pos: np.ndarray,
    mom: np.ndarray,
    isp: np.ndarray,
    *,
    action_mode: RLActionMode = "bernoulli",
) -> dict[str, Any]:
    obs = env.reset(pos, mom, isp)
    edge_logits = policy(obs)
    dist = torch.distributions.Bernoulli(logits=edge_logits)
    if action_mode == "threshold":
        edge_actions = (torch.sigmoid(edge_logits) > 0.5).float()
    else:
        edge_actions = dist.sample()
    mean_edge_entropy = dist.entropy().mean()
    partition_loss, cluster_labels = env.physics_for_edge_mask(edge_actions)
    baseline_loss = float(env._baseline_loss)
    reward = -partition_loss
    return {
        "obs": obs,
        "action": edge_actions,
        "reward": reward,
        "partition_loss": partition_loss,
        "baseline_loss": baseline_loss,
        "edge_entropy": mean_edge_entropy,
        "n_clusters": int(len(np.unique(cluster_labels))),
        "action_mode": action_mode,
        "event_pos": np.asarray(pos, dtype=np.float64, copy=True),
        "event_mom": np.asarray(mom, dtype=np.float64, copy=True),
        "event_isp": np.asarray(isp, copy=True),
    }

def train_reinforce(
    policy: GATAffinityPolicy,
    env: AffinityGraphEnv,
    event_sampler: EventSampler,
    *,
    optimizer: torch.optim.Optimizer,
    n_updates: int = 150,
    episodes_per_update: int = 8,
    lr_scheduler: LRScheduler | None = None,
    ent_coef: float = 0.02,
    max_grad_norm: float = 0.5,
    on_update: Callable[[dict[str, list]], None] | None = None,
    policy_coef: float = 1.0,
    rl_action_mode: RLActionMode = "bernoulli",
    val_events: list[tuple[np.ndarray, np.ndarray, np.ndarray]] | None = None,
    n_val_steps: int = 1,
    best_val_checkpoint_path: str | Path | None = None,
) -> dict[str, list]:
    history: dict[str, list] = {
        "episode_return": [],
        "partition_loss": [],
        "baseline_loss": [],
        "policy_loss": [],
        "return_baseline": [],
        "edge_entropy": [],
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
    val_ckpt: BestValPhysicsCheckpoint | None = (
        BestValPhysicsCheckpoint(best_val_checkpoint_path)
        if best_val_checkpoint_path is not None
        else None
    )
    pbar = tqdm(range(n_updates), desc="REINFORCE", miniters=1, mininterval=0.0, dynamic_ncols=True)
    for u in pbar:
        episode_returns: list[float] = []
        partition_losses: list[float] = []
        baseline_losses: list[float] = []
        cluster_counts: list[int] = []
        batch_observations: list[Data] = []
        batch_actions: list[torch.Tensor] = []

        for _ in range(episodes_per_update):
            pos, mom, isp = event_sampler()
            ep = collect_rollout(
                env,
                policy,
                pos,
                mom,
                isp,
                action_mode=rl_action_mode,
            )
            episode_returns.append(float(ep["reward"]))
            partition_losses.append(float(ep["partition_loss"]))
            baseline_losses.append(float(ep["baseline_loss"]))
            cluster_counts.append(ep["n_clusters"])
            batch_observations.append(ep["obs"])
            batch_actions.append(ep["action"])

        batch_mean_return = float(np.mean(episode_returns)) if episode_returns else 0.0
        history["return_baseline"].append(batch_mean_return)

        policy.train()
        optimizer.zero_grad()
        policy_loss_sum = entropy_sum = 0.0
        batch_size = max(len(batch_observations), 1)
        for i in range(len(batch_observations)):
            obs = batch_observations[i]
            logits = policy(obs)
            dist = torch.distributions.Bernoulli(logits=logits)
            stored_actions = batch_actions[i]
            mean_log_prob = dist.log_prob(stored_actions).mean()
            mean_entropy = dist.entropy().mean()
            episode_return = episode_returns[i]
            centered_return = torch.tensor(
                episode_return - batch_mean_return, dtype=torch.float32
            )
            policy_loss_term = -(mean_log_prob * centered_return)
            combined_loss = (policy_coef * policy_loss_term - ent_coef * mean_entropy) / batch_size
            combined_loss.backward()
            policy_loss_sum += policy_loss_term.item()
            entropy_sum += mean_entropy.item()
        nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)

        optimizer.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        if val_cfg and u % int(n_val_steps) == 0:
            validation_metrics = evaluate_validation_deterministic_policy(policy, env, val_events)
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
                    format_validation_console_line(
                        iter_key="update",
                        iter_value=int(u),
                        partition_loss_mev=validation_metrics["val_mean_partition_loss_mev"],
                        baseline_loss_mev=validation_metrics["val_mean_baseline_loss_mev"],
                        n_events=validation_metrics["val_n_events"],
                    )
                )
                if val_ckpt is not None:
                    val_ckpt.maybe_save(
                        validation_metrics["val_mean_partition_loss_mev"],
                        policy.state_dict(),
                        extra={"algorithm": "reinforce", "update": int(u)},
                    )

        if episode_returns:
            history["episode_return"].append(float(np.mean(episode_returns)))
        if partition_losses:
            history["partition_loss"].append(float(np.mean(partition_losses)))
        if baseline_losses and all(np.isfinite(baseline_losses)):
            history["baseline_loss"].append(float(np.mean(baseline_losses)))
        if cluster_counts:
            history["n_clusters"].append(float(np.mean(cluster_counts)))
        history["policy_loss"].append(policy_loss_sum / batch_size)
        history["edge_entropy"].append(entropy_sum / batch_size)

        progress_metrics: dict[str, float] = {}
        if partition_losses:
            progress_metrics["partition_gev"] = (
                float(np.mean(partition_losses)) / MEV_PER_GEV
            )
        if baseline_losses and all(np.isfinite(baseline_losses)):
            progress_metrics["baseline_gev"] = (
                float(np.mean(baseline_losses)) / MEV_PER_GEV
            )
        progress_metrics["policy_loss"] = float(policy_loss_sum / batch_size)
        progress_metrics["entropy"] = float(entropy_sum / batch_size)
        progress_metrics["lr"] = float(optimizer.param_groups[0]["lr"])
        progress_metrics["mean_return_gev"] = batch_mean_return / MEV_PER_GEV
        if episode_returns:
            progress_metrics["mean_episode_return_gev"] = (
                float(np.mean(episode_returns)) / MEV_PER_GEV
            )
        if cluster_counts:
            progress_metrics["mean_clusters"] = float(np.mean(cluster_counts))
        pbar.set_postfix(progress_metrics, refresh=True)
        if on_update is not None:
            on_update(history)

    return history
