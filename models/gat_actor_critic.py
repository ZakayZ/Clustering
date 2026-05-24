from typing import Any

import torch
import torch.nn as nn
from torch_geometric.data import Data

from models.constants import EDGE_PHYS_DIM
from models.policy import GATAffinityPolicy

class GATAffinityActorCritic(nn.Module):
    def __init__(
        self,
        policy: GATAffinityPolicy,
        *,
        value_mlp_hidden: int = 128,
        value_edge_hidden: int = 128,
        value_edge_embed_dim: int = 64,
    ) -> None:
        super().__init__()
        self.policy = policy
        d = int(policy.node_embed_dim)
        edge_in = 2 * d + EDGE_PHYS_DIM
        e_h = int(value_edge_hidden)
        e_out = int(value_edge_embed_dim)
        self.edge_value_encoder = nn.Sequential(
            nn.Linear(edge_in, e_h),
            nn.GELU(),
            nn.Linear(e_h, e_out),
        )
        self.value_head = nn.Sequential(
            nn.Linear(d + e_out, int(value_mlp_hidden)),
            nn.GELU(),
            nn.Linear(int(value_mlp_hidden), 1),
        )
        for seq in (self.edge_value_encoder, self.value_head):
            for m in seq.modules():
                if isinstance(m, nn.Linear):
                    nn.init.orthogonal_(m.weight, gain=1.0)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(
        self,
        data: Data,
        *,
        capture: dict[str, Any] | None = None,
        detach_value_features: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        logits, h = self.policy.forward_logits_and_h(data, capture=capture)
        h_v = h.detach() if detach_value_features else h
        h_pool = h_v.mean(dim=0)
        row, col = data.edge_index
        edge_repr = torch.cat(
            [h_v[row], h_v[col], data.edge_attr],
            dim=-1,
        )
        edge_emb = self.edge_value_encoder(edge_repr)
        if edge_emb.numel() == 0:
            edge_pool = torch.zeros(
                edge_emb.size(-1), device=h_v.device, dtype=h_v.dtype
            )
        else:
            edge_pool = edge_emb.mean(dim=0)
        x = torch.cat([h_pool, edge_pool], dim=-1)
        v = self.value_head(x)
        return logits, v.view(-1)[0]
