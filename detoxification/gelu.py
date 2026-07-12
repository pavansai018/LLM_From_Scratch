import torch
import torch.nn as nn
import config

class GELU(nn.Module):
    def __init__(self, cfg: dict = config.GPT2_SMALL_124M):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gelu_x: torch.Tensor = 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
        return gelu_x
