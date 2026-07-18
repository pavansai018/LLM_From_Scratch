import math
import torch
import torch.nn as nn

class LoRA(nn.Module):
    def __init__(self, original_linear: nn.Linear, rank: int, alpha: float, dropout: float=0.0):
        super().__init__()
        if rank <= 0:
            raise ValueError('Rank must be greater than 0')
        
        self.in_features: int = original_linear.in_features
        self.out_features: int = original_linear.out_features

        self.rank: int = rank
        self.alpha: float = alpha
        self.scaling: float = self.alpha / self.rank

        self.original_linear: nn.Linear = original_linear

        for parameter in self.original_linear.parameters():
            parameter.requires_grad = False

        '''
        A reduces input_dim -> rank
        weight shape: [rank, input_dim]
        '''
        self.lora_A = nn.Parameter(torch.empty(self.rank, self.in_features))

        '''
        B expands
        rank -> output_dim
        weight shape: [output_dim, rank]
        '''

        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank))

        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        '''
        x: [batch, seq_len, input_dim]

        original output: [batch, seq_len, output_dim]
        '''
        original_output = self.original_linear(x)

        # LoRA
        lora_input = self.dropout(x)

        '''
        apply A
        [batch, seq_len, input_dim] -> [batch, seq_len, rank]
        '''
        low_rank_features = nn.functional.linear(input=lora_input, weight=self.lora_A, bias=None)

        '''
        apply B
        [batch, seq_len, rank] -> [batch, seq_len, output_dim]
        '''
        lora_update = nn.functional.linear(input=low_rank_features, weight=self.lora_B, bias=None)

        lora_update = lora_update * self.scaling

        ouptut = original_output + lora_update

        return ouptut