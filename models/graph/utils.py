import torch

def policy_edges_from_directed(ei: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    src, dst = ei[0], ei[1]
    keep = src < dst
    return src[keep], dst[keep], torch.nonzero(keep, as_tuple=True)[0].long()
