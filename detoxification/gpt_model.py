import torch
import torch.nn as nn
import config
from transformer import TransformerBlock
from layer_norm import LayerNorm

class GPTModel(nn.Module):
    def __init__(self, cfg: dict = config.GPT2_SMALL_124M):
        super().__init__()

        self.token_embeddings: nn.Embedding = nn.Embedding(num_embeddings=cfg['vocab_size'], embedding_dim=cfg['emb_dim'])
        self.position_embeddings: nn.Embedding = nn.Embedding(num_embeddings=cfg['context_length'], embedding_dim=cfg['emb_dim'])

        self.dropout: nn.Dropout = nn.Dropout(p=cfg['drop_rate'])

        self.transformer_blocks: nn.Sequential = nn.Sequential(
            *[TransformerBlock(cfg=cfg) for _ in range(cfg['n_layers'])]
        )

        self.final_layer_norm: LayerNorm = LayerNorm(cfg=cfg)
        self.out_head: nn.Linear = nn.Linear(in_features=cfg['emb_dim'], out_features=cfg['vocab_size'], bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len = x.shape
        tok_emb = self.token_embeddings(x)
        pos_emb = self.position_embeddings(torch.arange(0, seq_len, device=x.device))

        x = tok_emb + pos_emb
        x = self.dropout(x)
        x = self.transformer_blocks(x)
        x = self.final_layer_norm(x)
        x = self.out_head(x)
        return x
    