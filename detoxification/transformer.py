import torch
import torch.nn as nn
import config
from layer_norm import LayerNorm
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward
from kv_cache import PastKeyValue

class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict = config.GPT2_SMALL_124M):
        super().__init__()

        self.norm1: LayerNorm = LayerNorm(cfg=cfg)
        self.multi_head_attention: MultiHeadAttention = MultiHeadAttention(cfg=cfg)
        self.dropout: nn.Dropout = nn.Dropout(p=cfg['drop_rate'])
        self.norm2: LayerNorm = LayerNorm(cfg=cfg)

        self.feed_forward: FeedForward = FeedForward(cfg=cfg)
        
    def forward(self, x: torch.Tensor, past_key_value: PastKeyValue | None = None, use_cache: bool = False) -> torch.Tensor | tuple[torch.Tensor, PastKeyValue]:
        skip_cell = x
        x = self.norm1(x)
        attention_result = self.multi_head_attention.forward(input_embeddings=x, pas_key_value=past_key_value, use_cache=use_cache)

        if use_cache:
            x, present_key_value = attention_result
        else:
            x = attention_result
        # x = self.multi_head_attention(x)
        x = self.dropout(x)

        x = x + skip_cell

        skip_cell = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + skip_cell

        if use_cache:
            return x, present_key_value

        return x
    