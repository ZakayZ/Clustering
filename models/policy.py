from typing import Any

import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data

from models.constants import EDGE_PHYS_DIM

class _MetaEdgeFeatureMLP(nn.Module):
    def __init__(self, mlp: nn.Module) -> None:
        super().__init__()
        self.mlp = mlp

    def forward(
        self,
        src: torch.Tensor,
        dst: torch.Tensor,
        edge_attr: torch.Tensor,
        u: torch.Tensor | None,
        batch: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.mlp(torch.cat([src, dst, edge_attr], dim=-1))

class GATAffinityPolicy(nn.Module):
    MAX_VALUE: float = 100.0

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        n_heads: int,
        n_gat_layers: int = 2,
        edge_mlp_depth: int = 1,
        edge_mlp_hidden: int | None = None,
    ) -> None:
        super().__init__()

        head_dim = hidden // n_heads
        h_e = edge_mlp_hidden if edge_mlp_hidden is not None else hidden
        edge_in = 2 * hidden + EDGE_PHYS_DIM
        channels = [edge_in] + [h_e] * edge_mlp_depth + [1]

        enc_layers: list[tuple[nn.Module, str]] = []
        in_ch = in_dim
        for layer_id in range(n_gat_layers):
            enc_layers.append(
                (
                    pyg_nn.GATv2Conv(
                        in_ch,
                        head_dim,
                        heads=n_heads,
                        concat=True,
                        edge_dim=EDGE_PHYS_DIM,
                        add_self_loops=False,
                        residual=(layer_id > 0),
                    ),
                    "x, edge_index, edge_attr -> x",
                )
            )
            enc_layers.append((nn.ELU(), "x -> x"))
            in_ch = hidden

        self.encoder = pyg_nn.Sequential("x, edge_index, edge_attr", enc_layers)
        edge_mlp = pyg_nn.models.MLP(
            channels,
            dropout=0.0,
            act="gelu",
            norm=None,
            plain_last=True,
        )
        self.edge_readout = pyg_nn.MetaLayer(
            edge_model=_MetaEdgeFeatureMLP(edge_mlp),
            node_model=None,
            global_model=None,
        )
        self.node_embed_dim: int = int(hidden)
        self._init_weights()

    def _init_weights(self) -> None:
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                gain = 0.01 if name.endswith(("lins.3", "lins.1")) else 1.0
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_logits_and_h(
        self, data: Data, *, capture: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = data.x
        ea = data.edge_attr
        if capture is not None:
            capture.clear()
            capture["n_nodes"] = int(data.x.size(0))
            capture["n_dir_edges"] = int(data.edge_index.size(1))
            capture["n_policy_edges"] = int(data.policy_edge_idx.numel())
            capture["node_x"] = x.detach()
            capture["edge_attr"] = ea.detach()
        h = self.encoder(x, data.edge_index, ea)
        if capture is not None:
            capture["h"] = h.detach()
        _, le, _ = self.edge_readout(h, data.edge_index, ea, u=None, batch=None)
        if capture is not None:
            capture["le_full"] = le.detach()
        raw = le[data.policy_edge_idx].view(-1)
        if capture is not None:
            capture["logits_raw"] = raw.detach()
        out = torch.nan_to_num(
            raw,
            nan=0.0,
            posinf=self.MAX_VALUE,
            neginf=-self.MAX_VALUE,
        ).clamp(
            -self.MAX_VALUE,
            self.MAX_VALUE,
        )
        if capture is not None:
            capture["logits_out"] = out.detach()
        return out, h

    def forward(self, data: Data, *, capture: dict[str, Any] | None = None) -> torch.Tensor:
        logits, _ = self.forward_logits_and_h(data, capture=capture)
        return logits

def init_policy_all_edges_off(policy: GATAffinityPolicy, logit_bias: float = -10.0) -> None:
    last = policy.edge_readout.edge_model.mlp.lins[-1]
    with torch.no_grad():
        last.weight.zero_()
        if last.bias is not None:
            last.bias.fill_(float(logit_bias))
