import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.eps: float = 1e-5
        self.scale: nn.Parameter = nn.Parameter(data=torch.ones(cfg['emb_dim']))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean_square = torch.pow(x, 2).mean(dim=-1, keepdim=True)
        rms = torch.rsqrt(mean_square)
        return self.scale * x * rms