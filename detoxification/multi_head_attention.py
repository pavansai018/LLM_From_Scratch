import torch
import torch.nn as nn
import config
from kv_cache import PastKeyValue

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: dict = config.GPT2_SMALL_124M):
        super().__init__()
        assert (cfg['emb_dim'] % cfg['num_heads'] == 0), 'emb_dim must be divisible by num_heads'
        self.emb_dim: int = cfg['emb_dim']
        self.context_length: int = cfg['context_length']
        self.num_heads: int = cfg['num_heads']
        self.head_dim: int = self.emb_dim // self.num_heads
        self.dropout: nn.Dropout = nn.Dropout(p=cfg['drop_rate'])
        self.qkv_bias: bool = cfg['qkv_bias']

        self.W_query: nn.Linear = nn.Linear(in_features=self.emb_dim, out_features=self.emb_dim, bias=self.qkv_bias)
        self.W_key: nn.Linear = nn.Linear(in_features=self.emb_dim, out_features=self.emb_dim, bias=self.qkv_bias)
        self.W_value: nn.Linear = nn.Linear(in_features=self.emb_dim, out_features=self.emb_dim, bias=self.qkv_bias)


        self.register_buffer(name='mask', tensor=torch.triu(input=torch.ones(size=(self.context_length, self.context_length)), diagonal=1), persistent=False)
        self.out_proj: nn.Linear = nn.Linear(in_features=self.emb_dim, out_features=self.emb_dim)

    def forward(self, input_embeddings: torch.Tensor, pas_key_value: PastKeyValue | None = None, use_cache: bool = False) -> torch.Tensor | tuple[torch.Tensor, PastKeyValue]:
        batch, num_tokens, d_in = input_embeddings.shape

        # [batch, num_tokens, emb_dim]
        queries: torch.Tensor = self.W_query(input_embeddings)
        keys: torch.Tensor = self.W_key(input_embeddings)
        values: torch.Tensor = self.W_value(input_embeddings)

        # [batch, num_tokens, num_heads, head_dim]
        queries = queries.view(batch, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(batch, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch, num_tokens, self.num_heads, self.head_dim)

        # [batch, num_heads, num_tokens, head_dim]
        queries = queries.transpose(dim0=1, dim1=2)
        keys = keys.transpose(dim0=1, dim1=2)
        values = values.transpose(dim0=1, dim1=2)

        past_length = 0
        if pas_key_value is not None:
            if not use_cache:
                raise ValueError('past_key_value requires use_cache=True')
            past_keys, past_values = pas_key_value
            past_length = past_keys.size(2)
            keys = torch.cat(tensors=(past_keys, keys), dim=2)
            values = torch.cat(tensors=(past_values, values), dim=2)


        '''
        queries: [batch, num_heads, num_tokens, head_dim]
        keys: [batch, num_heads, num_tokens, head_dim]
        keys.transpose: [batch, num_heads, head_dim, num_tokens]
        attention_scores: [batch, num_heads, num_tokens, num_tokens]
        '''
        attention_scores: torch.Tensor = queries @ keys.transpose(dim0=2, dim1=3)
        keys_length = keys.size(2)
        # mask: torch.Tensor = self.mask.bool()[:num_tokens, :num_tokens]
        mask: torch.Tensor = self.mask.bool()[past_length:past_length + num_tokens, :keys_length]
        attention_scores.masked_fill_(mask=mask, value=-torch.inf)
        attention_weights: torch.Tensor = torch.softmax(input=attention_scores/keys.shape[-1] ** 0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        '''
        context_vector: [batch, num_heads, num_tokens, num_tokens]
        values: [batch, num_heads, num_tokens, head_dim]
        context_vector: [batch, num_heads, num_tokens, head_dim]
        context_vector.transpose: [batch, num_tokens, num_heads, head_dim]
        '''
        context_vector: torch.Tensor = (attention_weights @ values).transpose(dim0=1, dim1=2)
        # [batch, num_tokens, emb_dim]
        context_vector = context_vector.contiguous().view(batch, num_tokens, d_in)

        output = self.out_proj(context_vector)
        if use_cache:
            present_key_value: PastKeyValue = (keys, values)
            return output, present_key_value
        return output
    

