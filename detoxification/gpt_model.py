import torch
import torch.nn as nn
import config
from transformer import TransformerBlock
from layer_norm import LayerNorm
from kv_cache import KVCache

class GPTModel(nn.Module):
    def __init__(self, cfg: dict = config.GPT2_SMALL_124M):
        super().__init__()

        self.context_length: int = cfg['context_length']
        self.num_layers: int = cfg['n_layers']
        self.token_embeddings: nn.Embedding = nn.Embedding(num_embeddings=cfg['vocab_size'], embedding_dim=cfg['emb_dim'])
        self.position_embeddings: nn.Embedding = nn.Embedding(num_embeddings=cfg['context_length'], embedding_dim=cfg['emb_dim'])

        self.dropout: nn.Dropout = nn.Dropout(p=cfg['drop_rate'])

        self.transformer_blocks: nn.Sequential = nn.Sequential(
            *[TransformerBlock(cfg=cfg) for _ in range(cfg['n_layers'])]
        )

        self.final_layer_norm: LayerNorm = LayerNorm(cfg=cfg)
        self.out_head: nn.Linear = nn.Linear(in_features=cfg['emb_dim'], out_features=cfg['vocab_size'], bias=False)

    def forward(self, x: torch.Tensor, kv_cache: KVCache | None = None, use_cache: bool = False) -> torch.Tensor | tuple[torch.Tensor, KVCache]:
        
        if use_cache and self.training:
            raise RuntimeError('KV cache can only used after model.eval()')
        if not use_cache and kv_cache is not None:
            raise ValueError('kv cache was provided while use_cache=False')
        if use_cache and kv_cache is None:
            kv_cache = KVCache(cfg=config.GPT2_SMALL_124M)
        if use_cache and len(kv_cache) != self.num_layers:
            raise ValueError('kv_cache cannot count does not match the model')
        
        past_length = kv_cache.sequence_length if use_cache else 0

        batch, seq_len = x.shape
        total_length = past_length + seq_len
        if total_length > self.context_length:
            raise ValueError(f'Total seq len {total_length} exceeds context length {self.context_length}')
        
        tok_emb = self.token_embeddings(x)
        # pos_emb = self.position_embeddings(torch.arange(0, seq_len, device=x.device))
        pos_emb = self.position_embeddings(torch.arange(start=past_length, end=total_length, device=x.device))
        x = tok_emb + pos_emb
        x = self.dropout(x)
        for layer_index, transformer_block in enumerate(self.transformer_blocks):
            if use_cache:
                past_key_value = kv_cache.get(layer_index)
                x, present_key_value = transformer_block.forward(x=x, past_key_value=past_key_value, use_cache=True)
                kv_cache.update(layer_index=layer_index, key_value=present_key_value)
            else:
                x = transformer_block(x)
        # x = self.transformer_blocks(x)
        x = self.final_layer_norm(x)
        x = self.out_head(x)
        if use_cache:
            return x, kv_cache
        return x
    