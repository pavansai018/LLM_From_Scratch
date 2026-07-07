import torch
import torch.nn as nn


class SimpleSelfAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, qkv_bias: bool = False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.W_query: torch.Tensor = nn.Linear(self.d_in, self.d_out, bias=qkv_bias)
        self.W_key: torch.Tensor = nn.Linear(self.d_in, self.d_out, bias=qkv_bias)
        self.W_value: torch.Tensor = nn.Linear(self.d_in, self.d_out, bias=qkv_bias)

        
    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        queries = self.W_query(input_embeddings)
        keys = self.W_key(input_embeddings)
        values = self.W_value(input_embeddings)

        attention_scores: torch.Tensor = queries @ keys.T
        attention_weights: torch.Tensor = torch.softmax(
            input=(attention_scores) / (keys.shape[1] ** 0.5), 
            dim=-1
        )
        context_vectors: torch.Tensor = attention_weights @ values
        return context_vectors

class CausalSelfAttention(nn.Module):
    def __init__(self,d_in: int, d_out: int, context_length: int, dropout: float = 0.0, qkv_bias: bool = False):
        super().__init__()
        self.d_in = d_in
        self.d_out = d_out
        self.dropout_rate = dropout
        self.context_length = context_length
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.register_buffer('mask', torch.triu(torch.ones(self.context_length, self.context_length), diagonal=1))


    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        b, num_tokens, d_in = input_embeddings.shape
        queries: torch.Tensor = self.W_query(input_embeddings)
        keys: torch.Tensor = self.W_key(input_embeddings)
        values: torch.Tensor = self.W_value(input_embeddings)

        attention_scores: torch.Tensor = queries @ keys.transpose(1, 2)
        attention_scores.masked_fill_(
            self.mask.bool()[: num_tokens, :num_tokens], -torch.inf
        )
        attention_weights = torch.softmax(
            attention_scores / (keys.shape[-1] ** 0.5), dim=-1
        )
        attention_weights = self.dropout(attention_weights)
        context_vector: torch.Tensor = attention_weights @ values

        return context_vector
    
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in: int, d_out: int, context_length: int, dropout: float, num_heads: int, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList(
            [
                CausalSelfAttention(d_in=d_in, d_out=d_out, context_length=context_length,) for _ in range(num_heads)
            ]
        )

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        return torch.cat([head(input_embeddings) for head in self.heads], dim=-1)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, context_length: int, num_heads: int, qkv_bias: bool = False, dropout: float = 0.0):
        super().__init__()
        assert (d_out % num_heads == 0), 'd_out must be divisible by num_heads'
        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.head_dim = self.d_out // self.num_heads
        
        self.W_query = nn.Linear(in_features=self.d_in, out_features=self.d_out, bias=qkv_bias)
        self.W_key = nn.Linear(in_features=self.d_in, out_features=self.d_out, bias=qkv_bias)
        self.W_value = nn.Linear(in_features=self.d_in, out_features=self.d_out, bias=qkv_bias)

        self.dropout = nn.Dropout(self.dropout_rate)
        self.out_projection = nn.Linear(in_features=self.d_out, out_features=self.d_out)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(self.context_length, self.context_length), diagonal=1)
        )

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        b, num_tokens, d_in = input_embeddings.shape

        queries: torch.Tensor = self.W_query(input_embeddings)
        keys: torch.Tensor = self.W_key(input_embeddings)
        values: torch.Tensor = self.W_value(input_embeddings)

        '''
        we implicitly split the matrix by adding num_heads dimension
        unroll the last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        '''
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # compute scaled dot-product (aka self-attention) with a causal mask
        attention_scores: torch.Tensor = queries @ keys.transpose(2, 3) # dot product for each head

        # original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        # use the mask to fill attention scores
        attention_scores.masked_fill_(mask_bool, -torch.inf)

        attention_weights: torch.Tensor = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        attention_weights: torch.Tensor = self.dropout(attention_weights)

        # shape: (b, num_tokens, num_heads, head_dim)
        context_vector: torch.Tensor = (attention_weights @ values).transpose(1, 2)

        # combine heads,  where self.d_out = self.num_heads * self.head_dim
        context_vector = context_vector.contiguous().view(b, num_tokens, self.d_out)
        context_vector = self.out_projection(context_vector) # optional projection

        return context_vector

class CausalMultiHeadAttention(nn.Module):
    def __init__(self, d_in: int, d_out: int, context_length: int, num_heads: int, dropout: float = 0.0, qkv_bias: bool = False):
        super().__init__()
        assert (d_out % num_heads == 0), 'd_out must be divisible by num_heads'
        self.d_in = d_in
        self.d_out = d_out
        self.context_length = context_length
        self.num_heads = num_heads
        self.dropout_rate = dropout
        self.qkv_bias = qkv_bias
        self.head_dim = self.d_out // self.num_heads

        self.dropout = nn.Dropout(self.dropout_rate)
        self.W_query: torch.Tensor = nn.Linear(in_features=self.d_in, out_features=self.d_out, bias=qkv_bias) # [batch, num_tokens, d_out]
        self.W_key: torch.Tensor = nn.Linear(in_features=self.d_in, out_features=self.d_out, bias=qkv_bias) # [batch, num_tokens, d_out]
        self.W_value: torch.Tensor = nn.Linear(in_features=self.d_in, out_features=self.d_out, bias=qkv_bias) # [batch, num_tokens, d_out]
        self.out_projection: torch.Tensor = nn.Linear(in_features=self.d_out, out_features=self.d_out)

        self.register_buffer('mask', torch.triu(torch.ones((self.context_length, self.context_length)), diagonal=1))

    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        b, num_tokens, d_in = input_embeddings.shape

        queries: torch.Tensor = self.W_query(input_embeddings) # [batch, num_tokens, d_out]
        keys: torch.Tensor = self.W_key(input_embeddings) # [batch, num_tokens, d_out]
        values: torch.Tensor = self.W_value(input_embeddings) # [batch, num_tokens, d_out]

        queries: torch.Tensor = queries.view(b, num_tokens, self.num_heads, self.head_dim)  # [batch, num_tokens, num_heads, head_dim]
        keys: torch.Tensor = keys.view(b, num_tokens, self.num_heads, self.head_dim) # [batch, num_tokens, num_heads, head_dim]
        values: torch.Tensor = values.view(b, num_tokens, self.num_heads, self.head_dim) # [batch, num_tokens, num_heads, head_dim]

        queries: torch.Tensor = queries.transpose(dim0=1, dim1=2) # [batch, num_heads, num_tokens, head_dim]
        keys: torch.Tensor = keys.transpose(dim0=1, dim1=2) # [batch, num_heads, num_tokens, head_dim]
        values: torch.Tensor = values.transpose(dim0=1, dim1=2) # [batch, num_heads, num_tokens, head_dim]

        # keys should become [batch, num_heads, head_dim, num_tokens] dot product on each head -> [batch, num_heads, num_tokens, num_tokens]
        attention_scores: torch.Tensor = queries @ keys.transpose(dim0=2, dim1=3)
        mask_bool: torch.Tensor = self.mask.bool()[:num_tokens, :num_tokens] # [num_tokens, num_tokens]
        attention_scores.masked_fill_(mask_bool, -torch.inf) # [batch, num_heads, num_tokens, num_tokens]
        attention_weights: torch.Tensor = torch.softmax(attention_scores / (keys.shape[-1] ** 0.5), dim=-1) # [batch, num_heads, num_tokens, num_tokens]
        attention_weights: torch.Tensor = self.dropout(attention_weights) # [batch, num_heads, num_tokens, num_tokens]

        # attention_weights = [batch, num_heads, num_tokens, num_tokens]
        # values = [batch, num_heads, num_tokens, head_dim]
        # attention x values = [batch, num_heads, num_tokens, head_dim]
        # transpose = [batch, num_tokens, num_heads, head_dim]
        context_vector: torch.Tensor = (attention_weights @ values).transpose(dim0=1, dim1=2) 
        context_vector: torch.Tensor = context_vector.contiguous().view(b, num_tokens, self.d_out) # [batch, num_tokens, d_out]

        return self.out_projection(context_vector) # [batch, num_tokens, d_out]



if __name__ == '__main__':
    inputs = torch.tensor(
        [
            [0.43, 0.15, 0.89],
            [0.55, 0.87, 0.66],
            [0.57, 0.85, 0.64],
            [0.22, 0.58, 0.33],
            [0.77, 0.25, 0.10],
            [0.05, 0.80, 0.55],
        ]
    )
    torch.manual_seed(123)
    inputs = torch.stack((inputs, inputs), dim=0)
    print(inputs.shape)

    batch_size, context_length, d_in = inputs.shape
    d_out = 4
    # print(SimpleSelfAttention(d_in=3, d_out=2).forward(inputs))
    # print(CausalSelfAttention(3, 2, 6).forward(inputs))
    # mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, num_heads=2, dropout=0.0)
    # print(mha.forward(inputs))
    # mha = MultiHeadAttention(d_in, d_out, context_length, num_heads=2, dropout=0.0)
    # print(mha.forward(inputs))
    mha = CausalMultiHeadAttention(d_in, d_out, context_length, num_heads=2, dropout=0.0)
    print(mha.forward(inputs))

