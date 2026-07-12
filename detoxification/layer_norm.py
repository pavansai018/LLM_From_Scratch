import torch
import torch.nn as nn
import config


class LayerNorm(nn.Module):
    def __init__(self, cfg: dict = config.GPT2_SMALL_124M):
        super().__init__()

        self.eps: float = 1e-5
        self.scale: nn.Parameter = nn.Parameter(data=torch.ones(cfg['emb_dim']))
        self.shift: nn.Parameter = nn.Parameter(data=torch.zeros(cfg['emb_dim']))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean: torch.Tensor = x.mean(dim=-1, keepdim=True)
        var: torch.Tensor = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x: torch.Tensor = (x - mean) / (torch.sqrt(var + self.eps))
        return self.scale * norm_x + self.shift
    