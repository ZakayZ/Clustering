"""Advantage actor–critic (A2C-style) training for affinity graph edge policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from torch_geometric.data import Data
from tqdm.auto import tqdm

from models import GATAffinityActorCritic
from models import AffinityGraphEnv, MEV_PER_GEV

from .utils import (
    RLActionMode,
    deterministic_eval_mean,
    _policy_grad_norm_by_prefix,
    _summarize_supervised_capture,
    _tensor_stats_f64,
    _total_grad_norm,
)

EventSampler = Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray]]


def collect_rollout_ac(
    env: AffinityGraphEnv,
    ac: GATAffinityActorCritic,
    pos: np.ndarray,
    mom: np.ndarray,
    isp: np.ndarray,
    *,
    action_mode: RLActionMode = "bernoulli",
    with_det_eval: bool = False,
) -> dict[str, Any]:
    """Actor–critic rollout: stores ``value`` = ``V(s)`` from :class:`GATAffinityActorCritic`."""
    obs = env.reset(pos, mom, isp)
    edge_logits, v = ac(obs)
    dist = torch.distributions.Bernoulli(logits=edge_logits)
    if action_mode == "threshold":
        a = (torch.sigmoid(edge_logits) > 0.5).float()
    else:
        a = dist.sample()
    ent = dist.entropy().mean()
    loss, labs = env.physics_for_edge_mask(a)
    l_base = float(env._baseline_loss)
    r = -loss
    out: dict[str, Any] = {
        "obs": obs,
        "action": a,
        "reward": r,
        "value": float(v.detach().item()),
        "partition_loss": loss,
        "baseline_loss": l_base,
        "edge_entropy": ent,
        "n_clusters": int(len(np.unique(labs))),
        "action_mode": action_mode,
        "event_pos": np.asarray(pos, dtype=np.float64, copy=True),
        "event_mom": np.asarray(mom, dtype=np.float64, copy=True),
        "event_isp": np.asarray(isp, copy=True),
    }
    if with_det_eval:
        if action_mode == "threshold":
            out["reward_det"] = float(r)
            out["partition_loss_det"] = float(loss)
            out["gap_det"] = float(loss - l_base)
            fm = float(a.mean().item())
            out["frac_edges_on_stoch"] = fm
            out["frac_edges_on_det"] = fm
        else:
            with torch.no_grad():
                a_det = (torch.sigmoid(edge_logits) > 0.5).float()
                loss_det, _ = env.physics_for_edge_mask(a_det)
            l_det = float(loss_det)
            out["reward_det"] = float(-l_det)
            out["partition_loss_det"] = l_det
            out["gap_det"] = float(l_det - l_base)
            out["frac_edges_on_stoch"] = float(a.mean().item())
            out["frac_edges_on_det"] = float(a_det.mean().item())
    return out


def warm_start_value_head(
    ac: GATAffinityActorCritic,
    env: AffinityGraphEnv,
    event_sampler: EventSampler,
    *,
    steps: int = 40,
    episodes_per_step: int = 8,
    lr: float = 3e-3,
    max_grad_norm: float = 1.0,
    rl_action_mode: RLActionMode = "threshold",
) -> None:
    """After supervised warm-start, fit only ``value_head`` to Monte Carlo returns (policy frozen).

    The critic is randomly initialized while edge logits are already trained; without this,
    ``R - V(s)`` advantages are dominated by critic error and destabilize actor–critic updates.
    """
    opt = torch.optim.Adam(ac.value_head.parameters(), lr=float(lr))
    for p in ac.policy.parameters():
        p.requires_grad_(False)
    ac.eval()
    try:
        for _ in range(max(1, int(steps))):
            opt.zero_grad()
            v_acc = 0.0
            n_e = 0
            for _i in range(max(1, int(episodes_per_step))):
                pos, mom, isp = event_sampler()
                with torch.no_grad():
                    ep = collect_rollout_ac(
                        env,
                        ac,
                        pos,
                        mom,
                        isp,
                        action_mode=rl_action_mode,
                    )
                    R = float(ep["reward"])
                    obs = ep["obs"]
                _, v_pred = ac(obs)
                R_t = torch.tensor(R, dtype=v_pred.dtype, device=v_pred.device)
                vf = 0.5 * (v_pred - R_t).pow(2)
                vf.backward()
                v_acc += float(vf.item())
                n_e += 1
            nn.utils.clip_grad_norm_(ac.value_head.parameters(), float(max_grad_norm))
            opt.step()
    finally:
        for p in ac.policy.parameters():
            p.requires_grad_(True)
        ac.train()


def train_actor_critic(
    ac: GATAffinityActorCritic,
    env: AffinityGraphEnv,
    event_sampler: EventSampler,
    *,
    optimizer: torch.optim.Optimizer,
    n_updates: int = 150,
    episodes_per_update: int = 8,
    lr_scheduler: LRScheduler | None = None,
    ent_coef: float = 0.02,
    value_coef: float = 0.5,
    max_grad_norm: float = 0.5,
    on_update: Callable[[dict[str, list]], None] | None = None,
    policy_coef: float = 1.0,
    center_adv: bool = True,
    diag_jsonl: Path | None = None,
    diag_every: int = 1,
    diag_det_rollouts: int = 32,
    rl_action_mode: RLActionMode = "bernoulli",
) -> dict[str, list]:
    """Advantage actor–critic with batch-centered advantages ``R - V(s)`` (optional ``center_adv``).

    Uses mean edge log-probability (same scaling as :func:`training.reinforce.train_reinforce`). Value target is
    Monte Carlo return ``R`` per one-shot episode. Gradient norms are clipped **separately** for
    ``ac.policy`` and ``ac.value_head`` so large policy gradients do not shrink the value-head step.
    """
    opt = optimizer
    diag_path = Path(diag_jsonl) if diag_jsonl is not None else None
    diag_f = diag_path.open("a", encoding="utf-8") if diag_path is not None else None
    history: dict[str, list] = {
        "episode_return": [],
        "partition_loss": [],
        "baseline_loss": [],
        "policy_loss": [],
        "value_loss": [],
        "value_mean": [],
        "advantage_mean": [],
        "edge_entropy": [],
        "n_clusters": [],
        "lr": [],
    }
    pbar = tqdm(range(n_updates), desc="A2C", miniters=1, mininterval=0.0, dynamic_ncols=True)
    try:
        for u in pbar:
            log_diag = diag_f is not None and diag_every > 0 and (u % diag_every == 0)
            ep_returns: list[float] = []
            part_losses: list[float] = []
            base_losses: list[float] = []
            n_clust: list[int] = []
            values_roll: list[float] = []
            batch_obs: list[Data] = []
            batch_act: list[torch.Tensor] = []
            rew_det_list: list[float] = []
            gap_det_list: list[float] = []
            fstoch_list: list[float] = []
            fdet_list: list[float] = []

            for _ in range(episodes_per_update):
                pos, mom, isp = event_sampler()
                ep = collect_rollout_ac(
                    env,
                    ac,
                    pos,
                    mom,
                    isp,
                    action_mode=rl_action_mode,
                    with_det_eval=log_diag,
                )
                ep_returns.append(float(ep["reward"]))
                part_losses.append(float(ep["partition_loss"]))
                base_losses.append(float(ep["baseline_loss"]))
                n_clust.append(ep["n_clusters"])
                values_roll.append(float(ep["value"]))
                batch_obs.append(ep["obs"])
                batch_act.append(ep["action"])
                if log_diag:
                    rew_det_list.append(float(ep["reward_det"]))
                    gap_det_list.append(float(ep["gap_det"]))
                    fstoch_list.append(float(ep["frac_edges_on_stoch"]))
                    fdet_list.append(float(ep["frac_edges_on_det"]))

            n_b = max(len(batch_obs), 1)
            raw_adv = [ep_returns[i] - values_roll[i] for i in range(len(batch_obs))]
            if center_adv and raw_adv:
                m_a = float(np.mean(raw_adv))
                adv_list = [float(a - m_a) for a in raw_adv]
            else:
                adv_list = [float(a) for a in raw_adv]

            ac.train()
            opt.zero_grad()
            pol_acc = ent_acc = v_acc = 0.0
            for i in range(len(batch_obs)):
                obs = batch_obs[i]
                logits, v_pred = ac(obs)
                dist = torch.distributions.Bernoulli(logits=logits)
                a = batch_act[i]
                logp = dist.log_prob(a).mean()
                ent = dist.entropy().mean()
                R = float(ep_returns[i])
                adv_t = torch.tensor(adv_list[i], dtype=torch.float32, device=logits.device)
                R_t = torch.tensor(R, dtype=torch.float32, device=logits.device)
                pol_t = -(logp * adv_t)
                v_t = 0.5 * (v_pred - R_t).pow(2)
                loss_t = (policy_coef * pol_t + value_coef * v_t - ent_coef * ent) / n_b
                loss_t.backward()
                pol_acc += pol_t.item()
                ent_acc += ent.item()
                v_acc += v_t.item()

            nn.utils.clip_grad_norm_(ac.policy.parameters(), max_grad_norm)
            nn.utils.clip_grad_norm_(ac.value_head.parameters(), max_grad_norm)
            grad_norm_total = _total_grad_norm(list(ac.parameters()))
            grad_by_pref = _policy_grad_norm_by_prefix(ac)
            lr_before_step = float(opt.param_groups[0]["lr"])

            if log_diag and diag_f is not None:
                ac.policy.eval()
                cap: dict[str, Any] = {}
                if batch_obs:
                    _ = ac.policy(batch_obs[0], capture=cap)
                    cap_sum = _summarize_supervised_capture(cap)
                    with torch.no_grad():
                        lo = ac.policy(batch_obs[0])
                        sig = torch.sigmoid(lo)
                    logits_stats = _tensor_stats_f64(lo)
                    sig_stats = _tensor_stats_f64(sig)
                else:
                    cap_sum = {}
                    logits_stats = {}
                    sig_stats = {}
                G_det_sweep, L_det_sweep, gap_det_sweep = deterministic_eval_mean(
                    ac.policy,
                    env,
                    event_sampler,
                    diag_det_rollouts,
                )
                ac.train()

                R_stoch = float(np.mean(ep_returns)) if ep_returns else float("nan")
                R_det_same = float(np.mean(rew_det_list)) if rew_det_list else float("nan")
                diag_row = {
                    "algo": "a2c",
                    "update": u,
                    "lr": lr_before_step,
                    "R_mean_stoch": R_stoch,
                    "R_mean_det_same_event": R_det_same,
                    "R_stoch_minus_R_det_same_event": R_stoch - R_det_same
                    if np.isfinite(R_stoch) and np.isfinite(R_det_same)
                    else None,
                    "L_pol_mean_stoch": float(np.mean(part_losses)) if part_losses else None,
                    "L_base_mean": float(np.mean(base_losses))
                    if base_losses and all(np.isfinite(base_losses))
                    else None,
                    "gap_mean_det_same_event": float(np.mean(gap_det_list))
                    if gap_det_list
                    else None,
                    "V_mean_rollout": float(np.mean(values_roll)) if values_roll else None,
                    "adv_mean": float(np.mean(adv_list)) if adv_list else None,
                    "adv_std": float(np.std(adv_list)) if len(adv_list) > 1 else 0.0,
                    "policy_loss_mean": float(pol_acc / n_b),
                    "value_loss_mean": float(v_acc / n_b),
                    "entropy_mean": float(ent_acc / n_b),
                    "grad_norm_total": grad_norm_total,
                    **grad_by_pref,
                    "logits": logits_stats,
                    "sigmoid": sig_stats,
                    "capture": cap_sum,
                    "det_eval_sweep_G_mean": G_det_sweep,
                    "det_eval_sweep_L_pol_mean": L_det_sweep,
                    "det_eval_sweep_gap_mean": gap_det_sweep,
                }
                diag_f.write(json.dumps(diag_row, default=str) + "\n")
                diag_f.flush()

            opt.step()
            if lr_scheduler is not None:
                lr_scheduler.step()
            history["lr"].append(float(opt.param_groups[0]["lr"]))

            if ep_returns:
                history["episode_return"].append(float(np.mean(ep_returns)))
            if part_losses:
                history["partition_loss"].append(float(np.mean(part_losses)))
            if base_losses and all(np.isfinite(base_losses)):
                history["baseline_loss"].append(float(np.mean(base_losses)))
            if n_clust:
                history["n_clusters"].append(float(np.mean(n_clust)))
            history["policy_loss"].append(pol_acc / n_b)
            history["value_loss"].append(v_acc / n_b)
            history["value_mean"].append(float(np.mean(values_roll)) if values_roll else 0.0)
            history["advantage_mean"].append(float(np.mean(adv_list)) if adv_list else 0.0)
            history["edge_entropy"].append(ent_acc / n_b)

            pf: dict[str, float] = {
                "pi": float(pol_acc / n_b),
                "Vloss": float(v_acc / n_b),
                "H": float(ent_acc / n_b),
                "lr": float(opt.param_groups[0]["lr"]),
            }
            if values_roll:
                pf["Vroll"] = float(np.mean(values_roll)) / MEV_PER_GEV
            if ep_returns:
                pf["G"] = float(np.mean(ep_returns)) / MEV_PER_GEV
            if part_losses:
                pf["L_pol"] = float(np.mean(part_losses)) / MEV_PER_GEV
            if base_losses and all(np.isfinite(base_losses)):
                pf["L_base"] = float(np.mean(base_losses)) / MEV_PER_GEV
            if n_clust:
                pf["n_cl"] = float(np.mean(n_clust))
            pbar.set_postfix(pf, refresh=True)
            if on_update is not None:
                on_update(history)
    finally:
        if diag_f is not None:
            diag_f.close()

    return history
