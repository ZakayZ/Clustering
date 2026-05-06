"""Supervised edge-BCE training against the spatial–momentum baseline (warm-start / standalone)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LRScheduler
from tqdm.auto import tqdm

from models import AffinityGraphEnv, GATAffinityPolicy, MEV_PER_GEV

from .utils import (
    EventSampler,
    baseline_edge_targets,
    weighted_bce_with_logits,
    _policy_grad_norm_by_prefix,
    _summarize_supervised_capture,
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
    supervised_debug: bool = False,
    supervised_debug_jsonl: Path | str | None = None,
) -> dict[str, list]:
    """BCE on edge logits against spatial–momentum baseline edge targets (warm-start / standalone).

    If ``lr_scheduler`` is set, ``lr_scheduler.step()`` runs after each ``optimizer.step()``.
    Set ``supervised_debug=True`` to record per-step forward statistics (first event in the
    outer batch), gradient norms before clipping, and target/logit summaries into
    ``history[\"supervised_debug\"]``; optionally append JSON lines to ``supervised_debug_jsonl``.
    """
    opt = optimizer or torch.optim.Adam(policy.parameters(), lr=lr)
    history: dict[str, list] = {
        "supervised_bce": [],
        "pretrain_partition_loss": [],
        "pretrain_baseline_loss": [],
        "pretrain_gap": [],
        "supervised_pos_weight": [],
        "pretrain_pos_recall_05": [],
        "pretrain_mean_prob_pos_edges": [],
        "pretrain_mean_prob_neg_edges": [],
        "pretrain_frac_pred_on_05": [],
        "pretrain_mean_n_clusters": [],
    }
    if supervised_debug:
        history["supervised_debug"] = []
    dbg_jsonl_path = Path(supervised_debug_jsonl) if supervised_debug_jsonl is not None else None
    policy.train()
    pbar_sup = tqdm(
        range(steps),
        desc="SupEdges",
        miniters=1,
        mininterval=0.0,
        dynamic_ncols=True,
    )
    for step_ix in pbar_sup:
        opt.zero_grad()
        sup_acc = 0.0
        sup_part: list[float] = []
        sup_base: list[float] = []
        n_eff = 0
        rec_acc = 0.0
        n_rec_batches = 0
        mp_pos_acc = 0.0
        n_mp_pos_batches = 0
        mp_neg_acc = 0.0
        n_mp_neg_batches = 0
        frac_on_acc = 0.0
        n_cl_acc = 0.0
        fc_for_debug: dict[str, Any] = {}
        dbg_logits: torch.Tensor | None = None
        dbg_tgt: torch.Tensor | None = None
        dbg_pw: float | None = None
        for mi in range(events_per_step):
            pos, mom, isp = event_sampler()
            obs = env.reset(pos, mom, isp)
            if supervised_debug and mi == 0:
                logits = policy(obs, capture=fc_for_debug)
            else:
                logits = policy(obs)
            edge_on = (torch.sigmoid(logits) > 0.5).float()
            l_pol, labs = env.physics_for_edge_mask(edge_on)
            sup_part.append(float(l_pol))
            sup_base.append(float(env._baseline_loss))
            tgt = baseline_edge_targets(env)
            if supervised_debug and mi == 0:
                dbg_logits = logits.detach().clone()
                dbg_tgt = tgt.detach().clone()
            loss_sup, pw = weighted_bce_with_logits(
                logits,
                tgt,
                auto_pos_weight=weighted_bce and pos_weight is None,
                pos_weight=pos_weight,
                pos_weight_power=pos_weight_power,
                max_pos_weight=pos_weight_max,
                focal_gamma=focal_gamma,
            )
            if supervised_debug and mi == 0:
                dbg_pw = float(pw)
            loss_sup.backward()
            sup_acc += float(loss_sup.item())
            history["supervised_pos_weight"].append(float(pw))
            n_eff += 1
            with torch.no_grad():
                prob = torch.sigmoid(logits)
                pos_m = tgt > 0.5
                neg_m = ~pos_m
                n_pos = int(pos_m.sum().item())
                if n_pos > 0:
                    tp = ((prob > 0.5) & pos_m).sum().item()
                    rec_acc += float(tp) / float(n_pos)
                    n_rec_batches += 1
                    mp_pos_acc += float(prob[pos_m].mean().item())
                    n_mp_pos_batches += 1
                n_neg = int(neg_m.sum().item())
                if n_neg > 0:
                    mp_neg_acc += float(prob[neg_m].mean().item())
                    n_mp_neg_batches += 1
                frac_on_acc += float((prob > 0.5).float().mean().item())
                n_cl_acc += float(len(np.unique(labs)))
        grad_prefix = dict(_policy_grad_norm_by_prefix(policy)) if supervised_debug else {}
        grad_norm_pre_clip = float(
            nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
        )
        opt.step()
        if lr_scheduler is not None:
            lr_scheduler.step()
        mean_sup = sup_acc / max(float(n_eff), 1.0)
        history["supervised_bce"].append(mean_sup)
        if sup_part:
            history["pretrain_partition_loss"].append(float(np.mean(sup_part)))
        if sup_base:
            history["pretrain_baseline_loss"].append(float(np.mean(sup_base)))
        if sup_part and sup_base:
            history["pretrain_gap"].append(float(np.mean(sup_part)) - float(np.mean(sup_base)))
        ne = max(float(n_eff), 1.0)
        history["pretrain_pos_recall_05"].append(
            rec_acc / max(float(n_rec_batches), 1.0)
        )
        history["pretrain_mean_prob_pos_edges"].append(
            mp_pos_acc / max(float(n_mp_pos_batches), 1.0)
        )
        history["pretrain_mean_prob_neg_edges"].append(
            mp_neg_acc / max(float(n_mp_neg_batches), 1.0)
        )
        history["pretrain_frac_pred_on_05"].append(frac_on_acc / ne)
        history["pretrain_mean_n_clusters"].append(n_cl_acc / ne)
        if supervised_debug and dbg_logits is not None and dbg_tgt is not None:
            dbg_row: dict[str, Any] = {
                "step": int(step_ix),
                "forward": _summarize_supervised_capture(fc_for_debug),
                "grad_norm_total_pre_clip": grad_norm_pre_clip,
                "outer_bce_mean": mean_sup,
                "pos_weight_first_batch": dbg_pw,
            }
            dbg_row.update(grad_prefix)
            with torch.no_grad():
                pos_m = dbg_tgt > 0.5
                prob = torch.sigmoid(dbg_logits)
                dbg_row["targets"] = {
                    "pos_rate": float(pos_m.float().mean()),
                    "n_pos": float(pos_m.sum().item()),
                    "mean_logit_pos": float(dbg_logits[pos_m].mean().item()) if pos_m.any() else 0.0,
                    "mean_logit_neg": float(dbg_logits[~pos_m].mean().item()) if (~pos_m).any() else 0.0,
                    "mean_prob_pos": float(prob[pos_m].mean().item()) if pos_m.any() else 0.0,
                    "mean_prob_neg": float(prob[~pos_m].mean().item()) if (~pos_m).any() else 0.0,
                }
            history["supervised_debug"].append(dbg_row)
            if dbg_jsonl_path is not None:
                dbg_jsonl_path.parent.mkdir(parents=True, exist_ok=True)
                with dbg_jsonl_path.open("a", encoding="utf-8") as jf:
                    jf.write(json.dumps(dbg_row) + "\n")
            tg = dbg_row["targets"]
            tqdm.write(
                f"[supervised_debug] step={step_ix} bce_out={mean_sup:.4g} "
                f"gn_pre_clip={grad_norm_pre_clip:.4g} "
                f"logit_pos/neg={tg['mean_logit_pos']:.3f}/{tg['mean_logit_neg']:.3f} "
                f"pos_rate={tg['pos_rate']:.4f}"
            )
        pf_sup: dict[str, float] = {"bce": mean_sup}
        if sup_part:
            pf_sup["L_pol"] = float(np.mean(sup_part)) / MEV_PER_GEV
        if sup_base:
            pf_sup["L_base"] = float(np.mean(sup_base)) / MEV_PER_GEV
        if sup_part and sup_base:
            pf_sup["gap"] = (float(np.mean(sup_part)) - float(np.mean(sup_base))) / MEV_PER_GEV
        if history["pretrain_pos_recall_05"]:
            pf_sup["posRec"] = float(history["pretrain_pos_recall_05"][-1])
        if history["pretrain_frac_pred_on_05"]:
            pf_sup["pOn"] = float(history["pretrain_frac_pred_on_05"][-1])
        pbar_sup.set_postfix(pf_sup, refresh=True)
        if on_update is not None:
            on_update(history)
    return history

