GPT2_SMALL_124M = {
    'vocab_size': 50257,
    'context_length': 1024,
    'emb_dim': 768,
    'num_heads': 12,
    'n_layers': 12,
    'drop_rate': 0.1,
    'qkv_bias': True,
    
}

GPT2_MEDIUM_355M = {
    'vocab_size': 50257,
    'context_length': 1024,
    'drop_rate': 0.0,
    'qkv_bias': True,
    'emb_dim': 1024,
    'num_heads': 16,
    'n_layers': 24,
}