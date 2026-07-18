import torch
import torch.nn as nn



class QLoRA(nn.Module):
    def __init__(self, quantized_weight: torch.Tensor, weight_scale: torch.Tensor, in_features: int, out_features: int, rank: int, alpha: float, bias: torch.Tensor | None = None):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank

        self.scaling = alpha / rank

        self.register_buffer('quantized_weight', quantized_weight)
        self.register_buffer('weight_scale', weight_scale)

        if bias is None:
            self.bias = None
        else:
            self.register_buffer('bias', bias)

        self.lora_A = nn.Parameter(torch.empty(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_weight = self.quantized_weight.to(x.dtype) * self.weight_scale.to(x.dtype)
        base_output = nn.functional.linear(input=x, weight=base_weight, bias=None if self.bias is None else self.bias.to(x.dtype))

        low_rank_features = nn.functional.linear(input=x, weight=self.lora_A, bias=None)
        lora_output = nn.functional.linear(input=low_rank_features, weight=self.lora_B, bias=None)

        lora_output = lora_output * self.scaling
        return base_output + lora_output
    