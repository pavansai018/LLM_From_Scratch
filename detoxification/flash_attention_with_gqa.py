import torch
import torch.nn as nn

class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()

        assert (cfg['emb_dim'] % cfg['n_heads'] == 0), 'emb_dim must be divisible by num_heads'
        assert (cfg['n_heads'] % cfg['kv_heads'] == 0), 'n_heads must be divisible by kv_heads'

        self.head_dim = cfg['emb_dim'] // cfg['n_heads']
        self.context_length = cfg['context_length']
        self.num_heads = cfg['n_heads']
        self.kv_heads = cfg['kv_heads']
        self.dropout = nn.Dropout(p=cfg['drop_rate'])


        self.W_query = nn.Linear(in_features=cfg['emb_dim'], out_features=cfg['emb_dim'], bias=cfg['qkv_bias'])
        self.W_key = nn.Linear(in_features=cfg['emb_dim'], out_features=self.kv_heads*self.head_dim, bias=cfg['qkv_bias'])
        self.W_value = nn.Linear(in_features=cfg['emb_dim'], out_features=self.kv_heads*self.head_dim, bias=cfg['qkv_bias'])


        self.register_buffer('mask', torch.triu(torch.ones((self.context_length, self.context_length)), diagonal=1), persistent=False)

        self.out_proj = nn.Linear(in_features=cfg['emb_dim'], out_features=cfg['emb_dim'])
    

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        batch, num_tokens, d_in = input_embeddings.shape

        # batch, num_tokens, emb_dim
        queries: torch.Tensor = self.W_query(input_embeddings)

        # batch, num_tokens, kv_heads*head_dim
        keys: torch.Tensor = self.W_key(input_embeddings)
        values: torch.Tensor = self.W_value(input_embeddings)

        # batch, num_tokens, num_heads, head_dim
        queries = queries.view(batch, num_tokens, self.num_heads, self.head_dim)
        

        # batch, num_heads, kv_heads, head_dim
        keys = keys.view(batch, num_tokens, self.kv_heads, self.head_dim)
        values = values.view(batch, num_tokens, self.kv_heads, self.head_dim)


        # batch, num_heads, num_tokens, head_dim
        queries = queries.transpose(dim0=1, dim1=2)

        # batch, kv_heads, num_tokens, head_dim
        keys = keys.transpose(dim0=1, dim1=2)
        values = values.transpose(dim0=1, dim1=2)

        # # batch, num_heads, num_tokens, head_dim
        # keys = keys.repeat_interleave(repeats=self.num_heads//self.kv_heads, dim=1)
        # values = values.repeat_interleave(repeats=self.num_heads//self.kv_heads, dim=1)

        # '''
        # queris: batch, num_heads, num_tokens, head_dim
        # keys.transpose: batch, num_heads, head_dim, num_tokens
        # attention_scores: batch, num_heads, num_tokens, num_tokens
        # '''
        # attention_scores = queries @ keys.transpose(dim0=2, dim1=3)
        # mask = self.mask.bool()[:num_tokens, :num_tokens]
        # attention_scores.masked_fill_(mask, -torch.inf)
        # attention_weights: torch.Tensor = torch.softmax(attention_scores/keys.shape[-1]**0.5, dim=-1)
        # attention_weights = self.dropout(attention_weights)

        # context_vector = (attention_weights @ values).transpose(dim0=1, dim1=2)
        context_vector = nn.functional.scaled_dot_product_attention(query=queries, key=keys, value=values, is_causal=True, enable_gqa=True, dropout_p=self.dropout.p)
        context_vector = context_vector.transpose(dim0=1, dim1=2)
        context_vector = context_vector.contiguous().view(batch, num_tokens, d_in)
        return self.out_proj(context_vector)
    