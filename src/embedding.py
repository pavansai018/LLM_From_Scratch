import torch
import torch.nn as nn
    

class InputEmbedding(nn.Module):
    def __init__(self, vocab_size: int, context_length: int, embedding_dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.context_length = context_length

        self.token_embedding = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim)
        self.position_embedding = nn.Embedding(num_embeddings=self.context_length, embedding_dim=self.embedding_dim)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        '''
        token_ids: [batch, sequence_length]
        output : [batch, sequence_length, embedding_dimension]
        '''
        B, T = token_ids.shape
        if T > self.context_length:
            raise ValueError( 
                f'Sequence length {T} exceeds block size {self.context_length}'
            )
        
        token_emb = self.token_embedding(token_ids) # [batch, seq_len, context_len]
        positions: torch.Tensor = torch.arange(0, T, device=token_ids.device, dtype=torch.long)

        pos_emb: torch.Tensor = self.position_embedding(positions) # [context_len, embedding_dimension]

        return token_emb + pos_emb.unsqueeze(0) # [batch, context_len, embedding_dimension]
    