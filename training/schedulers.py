from typing import TYPE_CHECKING

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, _enable_get_lr_call

if TYPE_CHECKING:
    from torch.optim import Optimizer

class CosineAnnealingWarmRestartsDecayMax(CosineAnnealingWarmRestarts):
    def __init__(
        self,
        optimizer: Optimizer,
        T_0: int,
        T_mult: int = 1,
        eta_min: float = 0.0,
        *,
        restart_eta_scale: float = 0.85,
        last_epoch: int = -1,
    ) -> None:
        super().__init__(optimizer, T_0, T_mult, eta_min, last_epoch)
        if restart_eta_scale <= 0:
            raise ValueError(f"restart_eta_scale must be positive, got {restart_eta_scale}")
        self.restart_eta_scale = float(restart_eta_scale)

    def step(self, epoch: int | float | None = None) -> None:
        if epoch is not None:
            super().step(epoch)
            return

        will_restart = self.T_cur + 1 >= self.T_i
        super().step()

        if not will_restart or self.restart_eta_scale == 1.0:
            return

        eta_floor = float(self.eta_min)
        self.base_lrs = [
            max(eta_floor, float(b) * self.restart_eta_scale) for b in self.base_lrs
        ]
        with _enable_get_lr_call(self):
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr(), strict=True):
                param_group["lr"] = lr
        self._last_lr = [group["lr"] for group in self.optimizer.param_groups]

__all__ = ["CosineAnnealingWarmRestartsDecayMax"]
