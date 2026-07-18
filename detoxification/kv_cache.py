from __future__ import annotations
import torch


PastKeyValue = tuple[torch.Tensor, torch.Tensor]

class KVCache:
    def __init__(self, cfg: dict):
        if cfg['n_layers'] <= 0:
            raise ValueError('num_layers must be >0')
        
        self._cache: list[PastKeyValue | None] = [None] * cfg['n_layers']

    def get(self, layer_index: int) -> PastKeyValue | None:
        return self._cache[layer_index]
    
    def update(self, layer_index: int, key_value: PastKeyValue) -> None:
        self._cache[layer_index] = key_value

    @property
    def sequence_length(self) -> int:
        first_layer = self._cache[0]
        if first_layer is None:
            return 0
        keys, _ = first_layer

        #[batch, num_heads, seq_len, head_dim]
        return keys.size(2)

    def clear(self) -> None:
        self._cache = [None] * len(self._cache)

    def __len__(self) -> int:
        return len(self._cache)
    
