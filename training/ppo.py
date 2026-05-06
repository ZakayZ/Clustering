"""Proximal Policy Optimization (PPO) for affinity graph edge policies."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from tqdm.auto import tqdm

from models import GATAffinityActorCritic
from models import AffinityGraphEnv, MEV_PER_GEV

from .a2c import warm_start_value_head
from .utils import (
    RLActionMode,
    deterministic_eval_mean,
    _policy_grad_norm_by_prefix,
    _summarize_supervised_capture,
    _tensor_stats_f64,
    _total_grad_norm,
)

EventSampler = Callable[[], tuple[np.ndarray, np.ndarray, np.ndarray]]


def collect_rollout_ppo(
    env: AffinityGraphEnv,
    ac: GATAffinityActorCritic,
    pos: np.ndarray,
    mom: np.ndarray,
    isp: np.ndarray,
    *,
    action_mode: RLActionMode = "bernoulli",
    with_det_eval: bool = False,
) -> dict[str, Any]:
    """PPO rollout: ``value``, ``old_log_prob`` (sum over edges, factorized Bernoulli), plus diagnostics."""
    obs = env.reset(pos, mom, isp)
    edge_logits, v = ac(obs)
    dist = torch.distributions.Bernoulli(logits=edge_logits)
    if action_mode == "threshold":
        a = (torch.sigmoid(edge_logits) > 0.5).float()
    else:
        a = dist.sample()
    ent = dist.entropy().mean()
    # Product policy over edges → log π(a|s) = Σ_i log π_i(a_i).
    old_log_prob = dist.log_prob(a).sum().detach()
    loss, labs = env.physics_for_edge_mask(a)
    l_base = float(env._baseline_loss)
    r = -loss
    out: dict[str, Any] = {
        "obs": obs,
        "action": a,
        "reward": r,
        "value": float(v.detach().item()),
        "old_log_prob": old_log_prob,
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


def train_ppo(
    ac: GATAffinityActorCritic,
    env: AffinityGraphEnv,
    event_sampler: EventSampler,
    *,
    optimizer: torch.optim.Optimizer,
    n_updates: int = 150,
    episodes_per_update: int = 8,
    ppo_epochs: int = 4,
    minibatch_size: int | None = None,
    clip_range: float = 0.2,
    value_clip_range: float | None = None,
    value_coef: float = 0.5,
    ent_coef: float = 0.02,
    policy_coef: float = 1.0,
    max_grad_norm: float = 0.5,
    normalize_advantage: bool = True,
    value_warmup_steps: int = 0,
    value_warmup_lr: float = 3e-3,
    value_warmup_episodes_per_step: int | None = None,
    lr_scheduler: LRScheduler | None = None,
    on_update: Callable[[dict[str, list]], None] | None = None,
    diag_jsonl: Path | None = None,
    diag_every: int = 1,
    diag_det_rollouts: int = 32,
    rl_action_mode: RLActionMode = "bernoulli",
) -> dict[str, list]:
    """Proximal Policy Optimization (Schulman et al.) on edge Bernoulli masks.

    Each environment step is one full graph: sample all edge bits, observe terminal reward
    ``R = -\\mathcal{L}_\\mathrm{pol}`` (MeV). Advantages are ``A = R - V(s)`` from rollout values
    (Monte Carlo / one-step TD). The policy ratio uses **sum** of per-edge log-probabilities
    (factorized Bernoulli). Several PPO epochs with minibatches reuse the same rollout buffer.

    Loss (per graph in a minibatch of size ``m``, averaged before ``backward``):

    - Clipped surrogate: ``-\\min(r A, \\mathrm{clip}(r,1-\\varepsilon,1+\\varepsilon) A)`` with
      ``r = \\exp(\\log\\pi_\\theta - \\log\\pi_{\\mathrm{old}})``.
    - Value: ``\\frac12 \\max((V-R)^2, (V^{\\mathrm{clip}}-R)^2)`` if ``value_clip_range`` is set,
      else ``\\frac12 (V-R)^2``. Default is no value clipping — returns here are MeV-scale
      (``R = -\\mathcal{L}_\\mathrm{pol}``); a small ``value_clip_range`` (e.g. ``0.2``) matches
      normalized environments and would stall the critic.
    - Entropy: mean edge entropy (same scale as :func:`training.reinforce.train_reinforce`).

    ``lr_scheduler.step()`` runs **once per outer update** after PPO optimization.

    Optional ``value_warmup_steps`` runs :func:`training.a2c.warm_start_value_head` once at the
    start so the critic matches MeV-scale returns before PPO updates.

    Gradient norms are clipped **separately** for ``ac.policy`` and ``ac.value_head`` so large
    policy gradients do not shrink the value-head step when a global clip is applied.
    """
    opt = optimizer
    diag_path = Path(diag_jsonl) if diag_jsonl is not None else None
    diag_f = diag_path.open("a", encoding="utf-8") if diag_path is not None else None
    if int(value_warmup_steps) > 0:
        ep_w = (
            int(value_warmup_episodes_per_step)
            if value_warmup_episodes_per_step is not None
            else int(episodes_per_update)
        )
        warm_start_value_head(
            ac,
            env,
            event_sampler,
            steps=int(value_warmup_steps),
            episodes_per_step=max(1, ep_w),
            lr=float(value_warmup_lr),
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
    pbar = tqdm(range(n_updates), desc="PPO", miniters=1, mininterval=0.0, dynamic_ncols=True)
    try:
        for u in pbar:
            log_diag = diag_f is not None and diag_every > 0 and (u % diag_every == 0)
            buffer: list[dict[str, Any]] = []
            ep_returns: list[float] = []
            part_losses: list[float] = []
            base_losses: list[float] = []
            n_clust: list[int] = []
            values_roll: list[float] = []
            rew_det_list: list[float] = []
            gap_det_list: list[float] = []
            fstoch_list: list[float] = []
            fdet_list: list[float] = []

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
                        with_det_eval=log_diag,
                    )
                buffer.append(ep)
                ep_returns.append(float(ep["reward"]))
                part_losses.append(float(ep["partition_loss"]))
                base_losses.append(float(ep["baseline_loss"]))
                n_clust.append(ep["n_clusters"])
                values_roll.append(float(ep["value"]))
                if log_diag:
                    rew_det_list.append(float(ep["reward_det"]))
                    gap_det_list.append(float(ep["gap_det"]))
                    fstoch_list.append(float(ep["frac_edges_on_stoch"]))
                    fdet_list.append(float(ep["frac_edges_on_det"]))

            B = len(buffer)
            n_b = max(B, 1)
            rewards_np = np.asarray([float(ep["reward"]) for ep in buffer], dtype=np.float64)
            values_np = np.asarray([float(ep["value"]) for ep in buffer], dtype=np.float64)
            adv_np = rewards_np - values_np
            if normalize_advantage and B > 1:
                adv_np = (adv_np - adv_np.mean()) / (float(adv_np.std()) + 1e-8)

            mb = int(minibatch_size) if minibatch_size is not None else B
            mb = max(1, min(mb, B))

            ac.train()
            dev = next(ac.parameters()).device
            dtype = next(ac.parameters()).dtype

            pol_acc = 0.0
            v_acc = 0.0
            ent_acc = 0.0
            kl_acc = 0.0
            clip_acc = 0.0
            n_opt_steps = 0
            grad_norm_total = 0.0
            grad_by_pref: dict[str, float] = {}

            for _ep in range(ppo_epochs):
                perm = np.random.permutation(B)
                for s in range(0, B, mb):
                    idxs = perm[s : s + mb]
                    m = max(len(idxs), 1)
                    opt.zero_grad()
                    loss_mb = torch.zeros((), device=dev, dtype=dtype)
                    for j in idxs:
                        ep = buffer[int(j)]
                        obs = ep["obs"]
                        a = ep["action"]
                        old_lp = ep["old_log_prob"]
                        if not isinstance(old_lp, torch.Tensor):
                            old_lp_t = torch.tensor(float(old_lp), device=dev, dtype=dtype)
                        else:
                            old_lp_t = old_lp.to(device=dev, dtype=dtype)
                        old_v = float(ep["value"])
                        ret = float(ep["reward"])
                        adv_j = float(adv_np[int(j)])

                        logits, v = ac(obs)
                        dist = torch.distributions.Bernoulli(logits=logits)
                        logp = dist.log_prob(a).sum()
                        ent = dist.entropy().mean()

                        adv_t = torch.tensor(adv_j, device=dev, dtype=dtype)
                        ret_t = torch.tensor(ret, device=dev, dtype=dtype)

                        ratio = torch.exp(logp - old_lp_t)
                        surr1 = ratio * adv_t
                        surr2 = (
                            torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range) * adv_t
                        )
                        pol_loss = -torch.min(surr1, surr2)

                        if value_clip_range is not None and float(value_clip_range) > 0.0:
                            vc = float(value_clip_range)
                            v_old_t = torch.tensor(old_v, device=dev, dtype=dtype)
                            v_clipped = v_old_t + torch.clamp(
                                v - v_old_t, -vc, vc
                            )
                            vf_loss = 0.5 * torch.max(
                                (v - ret_t).pow(2), (v_clipped - ret_t).pow(2)
                            )
                        else:
                            vf_loss = 0.5 * (v - ret_t).pow(2)

                        step_loss = (
                            policy_coef * pol_loss
                            + value_coef * vf_loss
                            - ent_coef * ent
                        ) / m
                        loss_mb = loss_mb + step_loss

                        with torch.no_grad():
                            pol_acc += float(pol_loss.item())
                            v_acc += float(vf_loss.item())
                            ent_acc += float(ent.item())
                            kl_acc += float((old_lp_t - logp).item())
                            off = torch.abs(ratio - 1.0) > clip_range
                            clip_acc += float(off.float().mean().item())
                            n_opt_steps += 1

                    loss_mb.backward()
                    nn.utils.clip_grad_norm_(ac.policy.parameters(), max_grad_norm)
                    nn.utils.clip_grad_norm_(ac.value_head.parameters(), max_grad_norm)
                    grad_norm_total = _total_grad_norm(list(ac.parameters()))
                    grad_by_pref = _policy_grad_norm_by_prefix(ac)
                    opt.step()

            lr_before_step = float(opt.param_groups[0]["lr"])

            if log_diag and diag_f is not None:
                ac.policy.eval()
                cap: dict[str, Any] = {}
                if buffer:
                    _ = ac.policy(buffer[0]["obs"], capture=cap)
                    cap_sum = _summarize_supervised_capture(cap)
                    with torch.no_grad():
                        lo = ac.policy(buffer[0]["obs"])
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
                    "algo": "ppo",
                    "update": u,
                    "lr": lr_before_step,
                    "ppo_epochs": ppo_epochs,
                    "minibatch_size": mb,
                    "clip_range": clip_range,
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
                    "adv_mean": float(np.mean(adv_np)) if B else None,
                    "adv_std": float(np.std(adv_np)) if B > 1 else 0.0,
                    "policy_loss_mean": pol_acc / max(n_opt_steps, 1),
                    "value_loss_mean": v_acc / max(n_opt_steps, 1),
                    "entropy_mean": ent_acc / max(n_opt_steps, 1),
                    "approx_kl_mean": kl_acc / max(n_opt_steps, 1),
                    "clip_fraction_mean": clip_acc / max(n_opt_steps, 1),
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
            history["policy_loss"].append(pol_acc / max(n_opt_steps, 1))
            history["value_loss"].append(v_acc / max(n_opt_steps, 1))
            history["value_mean"].append(float(np.mean(values_roll)) if values_roll else 0.0)
            history["advantage_mean"].append(float(np.mean(adv_np)) if B else 0.0)
            history["edge_entropy"].append(ent_acc / max(n_opt_steps, 1))
            history["approx_kl"].append(kl_acc / max(n_opt_steps, 1))
            history["clip_fraction"].append(clip_acc / max(n_opt_steps, 1))

            pf: dict[str, float] = {
                "pi": pol_acc / max(n_opt_steps, 1),
                "Vloss": v_acc / max(n_opt_steps, 1),
                "kl": kl_acc / max(n_opt_steps, 1),
                "clip": clip_acc / max(n_opt_steps, 1),
                "H": ent_acc / max(n_opt_steps, 1),
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
