"""TensorBoard scalar logging for training history dicts (notebook-friendly)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "TensorBoard logging requires the `tensorboard` package. "
        "Install with: pip install tensorboard"
    ) from e

from models import MEV_PER_GEV


class TensorBoardHistoryLogger:
    """Write the latest scalars from ``history`` each time an ``on_update`` fires."""

    def __init__(
        self,
        log_dir: str | Path,
        *,
        comment: str = "",
        flush_secs: int = 10,
    ) -> None:
        self._log_dir = Path(log_dir)
        self._writer = SummaryWriter(
            log_dir=str(self._log_dir),
            comment=comment,
            flush_secs=flush_secs,
        )

    def close(self) -> None:
        self._writer.close()

    def flush(self) -> None:
        self._writer.flush()

    @staticmethod
    def _latest_step(h: dict[str, list], primary_key: str) -> int:
        seq = h.get(primary_key)
        if not seq:
            return -1
        return len(seq) - 1

    def _w(self) -> Any:
        return self._writer

    def log_supervised(self, h: dict[str, list]) -> None:
        step = self._latest_step(h, "supervised_bce")
        if step < 0:
            return
        w = self._w()
        mev = float(MEV_PER_GEV)
        w.add_scalar("supervised/bce", float(h["supervised_bce"][-1]), step)
        if h.get("supervised_pos_weight"):
            w.add_scalar("supervised/pos_weight", float(h["supervised_pos_weight"][-1]), step)
        if h.get("pretrain_partition_loss"):
            w.add_scalar(
                "supervised/partition_loss_gev",
                float(h["pretrain_partition_loss"][-1]) / mev,
                step,
            )
        if h.get("pretrain_baseline_loss"):
            w.add_scalar(
                "supervised/baseline_loss_gev",
                float(h["pretrain_baseline_loss"][-1]) / mev,
                step,
            )
        if h.get("pretrain_gap"):
            w.add_scalar("supervised/gap_gev", float(h["pretrain_gap"][-1]) / mev, step)
        if h.get("pretrain_pos_recall_05"):
            w.add_scalar(
                "supervised/pos_recall_at_05",
                float(h["pretrain_pos_recall_05"][-1]),
                step,
            )
        if h.get("pretrain_frac_pred_on_05"):
            w.add_scalar(
                "supervised/frac_edges_pred_on_at_05",
                float(h["pretrain_frac_pred_on_05"][-1]),
                step,
            )
        if h.get("pretrain_mean_prob_pos_edges"):
            w.add_scalar(
                "supervised/mean_prob_pos_edges",
                float(h["pretrain_mean_prob_pos_edges"][-1]),
                step,
            )
        if h.get("pretrain_mean_prob_neg_edges"):
            w.add_scalar(
                "supervised/mean_prob_neg_edges",
                float(h["pretrain_mean_prob_neg_edges"][-1]),
                step,
            )
        if h.get("pretrain_mean_n_clusters"):
            w.add_scalar(
                "supervised/mean_n_clusters",
                float(h["pretrain_mean_n_clusters"][-1]),
                step,
            )

    def log_reinforce(self, h: dict[str, list]) -> None:
        step = self._latest_step(h, "episode_return")
        if step < 0:
            return
        w = self._w()
        mev = float(MEV_PER_GEV)
        w.add_scalar("reinforce/episode_return_gev", float(h["episode_return"][-1]) / mev, step)
        if h.get("return_baseline"):
            w.add_scalar(
                "reinforce/return_baseline_gev",
                float(h["return_baseline"][-1]) / mev,
                step,
            )
        if h.get("partition_loss"):
            w.add_scalar(
                "reinforce/partition_loss_gev",
                float(h["partition_loss"][-1]) / mev,
                step,
            )
        if h.get("baseline_loss"):
            w.add_scalar(
                "reinforce/baseline_loss_gev",
                float(h["baseline_loss"][-1]) / mev,
                step,
            )
        if h.get("policy_loss"):
            w.add_scalar("reinforce/policy_loss", float(h["policy_loss"][-1]), step)
        if h.get("lr"):
            w.add_scalar("reinforce/lr", float(h["lr"][-1]), step)
        if h.get("edge_entropy"):
            w.add_scalar("reinforce/edge_entropy", float(h["edge_entropy"][-1]), step)
        if h.get("n_clusters"):
            w.add_scalar("reinforce/mean_n_clusters", float(h["n_clusters"][-1]), step)

    def log_actor_critic(self, h: dict[str, list], *, algo: str) -> None:
        """Log ``train_actor_critic`` / ``train_ppo`` history. ``algo`` is ``a2c`` or ``ppo``."""
        step = self._latest_step(h, "episode_return")
        if step < 0:
            return
        w = self._w()
        mev = float(MEV_PER_GEV)
        p = algo.strip().lower()
        if p not in ("a2c", "ppo"):
            raise ValueError(f"algo must be 'a2c' or 'ppo', got {algo!r}")
        w.add_scalar(f"{p}/episode_return_gev", float(h["episode_return"][-1]) / mev, step)
        if h.get("value_mean"):
            w.add_scalar(f"{p}/value_mean_gev", float(h["value_mean"][-1]) / mev, step)
        if h.get("advantage_mean"):
            w.add_scalar(f"{p}/advantage_mean", float(h["advantage_mean"][-1]), step)
        if h.get("partition_loss"):
            w.add_scalar(f"{p}/partition_loss_gev", float(h["partition_loss"][-1]) / mev, step)
        if h.get("baseline_loss"):
            w.add_scalar(f"{p}/baseline_loss_gev", float(h["baseline_loss"][-1]) / mev, step)
        if h.get("policy_loss"):
            w.add_scalar(f"{p}/policy_loss", float(h["policy_loss"][-1]), step)
        if h.get("value_loss"):
            w.add_scalar(f"{p}/value_loss", float(h["value_loss"][-1]), step)
        if h.get("lr"):
            w.add_scalar(f"{p}/lr", float(h["lr"][-1]), step)
        if h.get("edge_entropy"):
            w.add_scalar(f"{p}/edge_entropy", float(h["edge_entropy"][-1]), step)
        if h.get("n_clusters"):
            w.add_scalar(f"{p}/mean_n_clusters", float(h["n_clusters"][-1]), step)
        if p == "ppo":
            if h.get("approx_kl"):
                w.add_scalar("ppo/approx_kl", float(h["approx_kl"][-1]), step)
            if h.get("clip_fraction"):
                w.add_scalar("ppo/clip_fraction", float(h["clip_fraction"][-1]), step)
