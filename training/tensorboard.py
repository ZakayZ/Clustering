import shutil
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from models import MEV_PER_GEV

_GEV = float(MEV_PER_GEV)

def clear_tensorboard_notebook_root(notebook_root: str | Path) -> None:
    root = Path(notebook_root)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)

class TensorBoardHistoryLogger:
    def __init__(
        self,
        log_dir: str | Path,
        *,
        comment: str = "",
        flush_secs: int = 10,
        clean: bool = True,
    ) -> None:
        self._log_dir = Path(log_dir).resolve()
        if clean and self._log_dir.exists():
            shutil.rmtree(self._log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = SummaryWriter(
            log_dir=str(self._log_dir),
            comment=comment,
            flush_secs=flush_secs,
        )
        self._val_logged = 0

    def close(self) -> None:
        self.flush()
        self._writer.close()

    def flush(self) -> None:
        self._writer.flush()

    @staticmethod
    def _step(h: dict[str, list], key: str) -> int:
        seq = h.get(key)
        return len(seq) - 1 if seq else -1

    def _emit(
        self,
        h: dict[str, list],
        prefix: str,
        step: int,
        keys_tags_gev: list[tuple[str, str, bool]],
    ) -> None:
        w = self._writer
        for hk, tag, as_gev in keys_tags_gev:
            seq = h.get(hk)
            if not seq:
                continue
            v = float(seq[-1])
            if as_gev:
                v /= _GEV
            w.add_scalar(f"{prefix}/{tag}", v, step)

    def _sync_validation(self, h: dict[str, list], prefix: str) -> None:
        steps = h.get("val_step") or []
        losses = h.get("val_mean_partition_loss_mev") or []
        w = self._writer
        pfx = prefix.strip().lower()
        while self._val_logged < len(steps) and self._val_logged < len(losses):
            w.add_scalar(
                f"{pfx}/val/partition_loss_gev",
                float(losses[self._val_logged]) / _GEV,
                int(steps[self._val_logged]),
            )
            self._val_logged += 1

    def log_supervised(self, h: dict[str, list]) -> None:
        step = self._step(h, "supervised_bce")
        if step < 0:
            return
        self._emit(
            h,
            "supervised",
            step,
            [
                ("pretrain_partition_loss", "partition_loss_gev", True),
                ("pretrain_baseline_loss", "baseline_loss_gev", True),
                ("supervised_bce", "bce", False),
                ("supervised_pos_weight", "pos_weight", False),
                ("pretrain_pos_recall_05", "pos_recall_at_05", False),
                ("pretrain_frac_pred_on_05", "frac_edges_pred_on_at_05", False),
                ("pretrain_mean_prob_pos_edges", "mean_prob_pos_edges", False),
                ("pretrain_mean_prob_neg_edges", "mean_prob_neg_edges", False),
                ("pretrain_mean_n_clusters", "mean_n_clusters", False),
            ],
        )
        self._sync_validation(h, "supervised")

    def log_reinforce(self, h: dict[str, list]) -> None:
        step = self._step(h, "episode_return")
        if step < 0:
            return
        self._emit(
            h,
            "reinforce",
            step,
            [
                ("partition_loss", "partition_loss_gev", True),
                ("baseline_loss", "baseline_loss_gev", True),
                ("episode_return", "episode_return_gev", True),
                ("return_baseline", "return_baseline_gev", True),
                ("policy_loss", "policy_loss", False),
                ("lr", "lr", False),
                ("edge_entropy", "edge_entropy", False),
                ("n_clusters", "mean_n_clusters", False),
            ],
        )
        self._sync_validation(h, "reinforce")

    def log_actor_critic(self, h: dict[str, list], *, algo: str) -> None:
        step = self._step(h, "episode_return")
        if step < 0:
            return
        p = algo.strip().lower()
        if p not in ("a2c", "ppo"):
            raise ValueError(f"algo must be 'a2c' or 'ppo', got {algo!r}")
        base = [
            ("partition_loss", "partition_loss_gev", True),
            ("baseline_loss", "baseline_loss_gev", True),
            ("episode_return", "episode_return_gev", True),
            ("value_mean", "value_mean_gev", True),
            ("advantage_mean", "advantage_mean", False),
            ("policy_loss", "policy_loss", False),
            ("value_loss", "value_loss", False),
            ("lr", "lr", False),
            ("edge_entropy", "edge_entropy", False),
            ("n_clusters", "mean_n_clusters", False),
        ]
        self._emit(h, p, step, base)
        if p == "ppo":
            self._emit(
                h,
                "ppo",
                step,
                [
                    ("approx_kl", "approx_kl", False),
                    ("clip_fraction", "clip_fraction", False),
                ],
            )
        self._sync_validation(h, p)
