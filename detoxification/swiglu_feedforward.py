import torch
import torch.nn as nn

class SWiGLU_FeedForward(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()

        self.linear1: nn.Linear = nn.Linear(in_features=cfg['emb_dim'], out_features=8*cfg['emb_dim'] // 3, bias=False)
        self.linear2: nn.Linear = nn.Linear(in_features=cfg['emb_dim'], out_features=8*cfg['emb_dim'] // 3, bias=False)
        self.linear3: nn.Linear = nn.Linear(in_features=8*cfg['emb_dim'] // 3, out_features=cfg['emb_dim'], bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a = self.linear1(x)
        b = self.linear2(x)

        hidden = x * torch.sigmoid(a) * b

        return self.linear3(hidden)