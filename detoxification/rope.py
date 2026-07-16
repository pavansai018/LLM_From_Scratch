import torch
import torch.nn as nn

class RotaryEmbedding(nn.Module):
    '''
    Applies RoPE to queries or keys
    Input shape:
        [batch, num_heads, seq_len, head_dim]

    Output shape:
        [batch, num_heads, seq_len, head_dim]

    '''
    def __init__(self, head_dim: int, base: float = 10_000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError('head dim must be even for rope')
        self.head_dim = head_dim

        '''
        RoPE process dimensions as pairs
        [dim0, dim1]
        [dim2, dim3]
        [dim4, dim5]

        therefore we need one frequency for each pair

        for head_dim=8:
        pair_start_indices=[0, 2, 4, 8]
        '''
        pair_start_indices = torch.arange(start=0, end=head_dim, step=2, dtype=torch.float32)

        '''
        each dimension pair receives a different rotation speed
        shape: [head_dim/2]
        '''
        inverse_frequencies = 1.0 / (base ** (pair_start_indices) / head_dim)

        # frequencies are fixed constants. no need to be learned during training
        self.register_buffer(
            name='inverse_frequencies',
            tensor=inverse_frequencies,
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(
                f'expected x with shape: [batch, num_heads, num_tokens, head_dim]'
            )
        
        if x.size(-1) != self.head_dim:
            raise ValueError(
                f'Expected head_dim: {self.head_dim}, but received head_dim={x.size(-1)}'
            )
        
        sequence_length = x.size(2)

        '''
        token positions
        [0, 1, 2, 3...., seq_len-1]
        shape: [seq_len]
        '''
        positions = torch.arange(start=0, end=sequence_length, device=x.device, dtype=torch.float32)


        '''
        calculate one rotation angle for every:
        token position x dimension_pair
        positions[:, None] = [seq_len, 1]
        inv_freq[None:, :] = [1, head_dim/2]

        angles: [seq_len, head_dim/2]
        
        '''
        angles = positions[:, None] * self.inverse_frequencies[None, :]

        '''
        calculate cosine and sine needed by 2d rotation formula
        initial shape: [seq_len, head_dim/2]
        final shape: [1, 1, seq_length, head_dim/2]
        
        the leading dimensions allow the same positional rotations to be used for every batch and every head

        '''
        cos_values = torch.cos(angles)[None, None, :, :]
        sin_values = torch.sin(angles)[None, None, :, :]

        cos_values = cos_values.to(dtype=x.dtype)
        sin_values = sin_values.to(dtype=x.dtype)

        '''
        example:

        x = [10, 20, 30, 40, 50, 60]
        pair_first = [10, 30, 50]
        pair_second = [20, 40, 60]

        these represent original pairs [10,20], [20,30], [40, 50]
        '''
        pair_first = x[..., 0::2]
        pair_second = x[..., 1::2]

        '''
        rotate each pair
        new_first = first * cos - second * sin
        new_second = first * sin + second * cos
        '''
        rotated_pair_first = pair_first * cos_values - pair_second * sin_values
        rotated_pair_second = pair_first * sin_values + pair_second * cos_values

        output = torch.empty_like(x)
        output[..., 0::2] = rotated_pair_first
        output[..., 1::2] = rotated_pair_second

        return output
    

if __name__ == '__main__':
    rope = RotaryEmbedding(64)
    a = torch.rand(1024)
    a = a.view(1, 1, 1024//64, 64)
    print(a.shape)
    o = rope.forward(a)
    print(o.shape)