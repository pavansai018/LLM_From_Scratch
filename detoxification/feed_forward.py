import torch
import torch.nn as nn
import config
from gelu import GELU

class FeedForward(nn.Module):
    def __init__(self, cfg: dict = config.GPT2_SMALL_124M):
        super().__init__()

        self.layers: nn.Sequential = nn.Sequential(
            nn.Linear(in_features=cfg['emb_dim'], out_features=4*cfg['emb_dim']),
            GELU(),
            nn.Linear(in_features=4*cfg['emb_dim'], out_features=cfg['emb_dim'])

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
