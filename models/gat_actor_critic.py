"""Policy + scalar value head (mean-pooled GAT embeddings) for A2C / PPO."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torch_geometric.data import Data

from models.policy import GATAffinityPolicy


class GATAffinityActorCritic(nn.Module):
    """Policy (:class:`~models.policy.GATAffinityPolicy` edge logits) + scalar ``V(s)``.

    ``V(s)`` is produced from **mean-pooled** GAT node embeddings (permutation-invariant graph
    summary) plus a small MLP — one forward pass for both heads. Policy and value losses both
    backprop through the GAT encoder (joint actor–critic trunk).
    """

    def __init__(
        self,
        policy: GATAffinityPolicy,
        *,
        value_mlp_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.policy = policy
        d = int(policy.node_embed_dim)
        self.value_head = nn.Sequential(
            nn.Linear(d, int(value_mlp_hidden)),
            nn.GELU(),
            nn.Linear(int(value_mlp_hidden), 1),
        )
        for lin in self.value_head:
            if isinstance(lin, nn.Linear):
                nn.init.orthogonal_(lin.weight, gain=1.0)
                if lin.bias is not None:
                    nn.init.zeros_(lin.bias)

    def forward(
        self,
        data: Data,
        *,
        capture: dict[str, Any] | None = None,
        detach_value_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return edge logits and scalar ``V(s)`` (full gradient path through the encoder).

        If ``detach_value_features`` is True, only the value head backprops w.r.t. its input;
        gradients from the value loss **do not** flow into the GAT encoder. Policy and entropy
        terms still train the encoder via ``logits``. Use this in PPO/A2C when large MeV-scale
        critic error would otherwise overwrite good policy features.
        """
        logits, h = self.policy.forward_logits_and_h(data, capture=capture)
        h_pool = h.mean(dim=0)
        src = h_pool.detach() if detach_value_features else h_pool
        v = self.value_head(src)
        return logits, v.view(-1)[0]
