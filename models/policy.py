"""GAT policy for affinity edges."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data

from models.constants import EDGE_PHYS_DIM


class _MetaEdgeFeatureMLP(nn.Module):
    """Maps MetaLayer's ``(src, dst, edge_attr)`` to a single tensor for the edge MLP."""

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
    """GATv2 encoder + MetaLayer edge MLP; logits are one per undirected kNN edge (``policy_edge_idx``)."""

    MAX_VALUE: float = 100.0

    def __init__(
        self,
        in_dim: int,
        hidden: int,
        n_heads: int,
        n_gat_layers: int = 2,
        edge_mlp_depth: int = 1,
        edge_mlp_hidden: int | None = None,
        running_norm: bool = True,
        running_norm_momentum: float = 0.1,
    ) -> None:
        super().__init__()

        # Per-feature BatchNorm1d (affine=False → no learnable γ/β).
        # N nodes / E edges per event serve as the "batch" dimension, so BatchNorm
        # normalises each feature across all nodes/edges in the current graph and
        # maintains running mean/var across events (momentum=running_norm_momentum).
        # momentum=0.1 matches PyTorch default for running-stat EMA during train().
        # During model.eval() (RL rollouts) the frozen running stats are used — safe
        # for RL because the normalisation is a fixed function at inference time.
        # Set running_norm=False to disable entirely.
        bn_kw = dict(affine=False, momentum=running_norm_momentum)
        self.node_norm = nn.BatchNorm1d(in_dim,        **bn_kw) if running_norm else nn.Identity()
        self.edge_norm = nn.BatchNorm1d(EDGE_PHYS_DIM, **bn_kw) if running_norm else nn.Identity()

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
        # Width of the last GAT stack (``n_heads * head_dim``); used by PPO value head on pooled ``h``.
        self.node_embed_dim: int = int(hidden)
        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal init for linear layers; output layer scaled small for near-zero initial logits.

        Orthogonal init (Saxe et al. 2013) keeps gradient norms stable across layers without
        requiring LayerNorm. It is standard in PPO implementations and speeds up early convergence.
        """
        for name, m in self.named_modules():
            if isinstance(m, nn.Linear):
                gain = 0.01 if name.endswith(("lins.3", "lins.1")) else 1.0
                nn.init.orthogonal_(m.weight, gain=gain)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward_logits_and_h(
        self, data: Data, *, capture: dict[str, Any] | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Policy edge logits and GAT node embeddings ``h`` (same forward as :meth:`forward`).

        Used by :mod:`models.gat_actor_critic` for the value head on pooled ``h``.
        """
        x = self.node_norm(data.x)
        ea = self.edge_norm(data.edge_attr)
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
    """Make initial edge logits almost constant and negative so rollouts are ~all edges off.

    Zeros the final linear weights so logits equal the bias (no dependence on GAT features);
    bias defaults to ``-10`` (sigmoid ≈ 4.5e-5). Use for unsupervised REINFORCE cold starts.
    """

    last = policy.edge_readout.edge_model.mlp.lins[-1]
    with torch.no_grad():
        last.weight.zero_()
        if last.bias is not None:
            last.bias.fill_(float(logit_bias))
