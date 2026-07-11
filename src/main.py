import torch
import tiktoken
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

class LayerNorm(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.eps = 1e-5
        self.scale: nn.Parameter = nn.Parameter(data=torch.ones(cfg['emb_dim']))
        self.shift: nn.Parameter = nn.Parameter(data=torch.zeros(cfg['emb_dim']))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x = (x - mean) / (torch.sqrt(var + self.eps))
        return self.scale * norm_x + self.shift

class GELU(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gelu_x: torch.Tensor = 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0/torch.pi)) * (x + 0.044715*torch.pow(x, 3))))
        return gelu_x

class FeedForward(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()

        self.layers: nn.Sequential = nn.Sequential(
            nn.Linear(in_features=cfg['emb_dim'], out_features=4*cfg['emb_dim']),
            GELU(),
            nn.Linear(in_features=4*cfg['emb_dim'], out_features=cfg['emb_dim']),

        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        assert (cfg['emb_dim'] % cfg['num_heads'] == 0), 'emb_dim must be divisible by num_heads'
        self.num_heads: int = cfg['num_heads']
        self.emb_dim: int = cfg['emb_dim']
        self.head_dim: int = self.emb_dim // self.num_heads
        self.dropout: nn.Dropout = nn.Dropout(p=cfg['drop_rate'])
        self.context_length: int = cfg['context_length']
        self.qkv_bias: bool = cfg['qkv_bias']

        self.W_query: nn.Linear = nn.Linear(in_features=self.emb_dim, out_features=self.emb_dim, bias=self.qkv_bias)
        self.W_key: nn.Linear = nn.Linear(in_features=self.emb_dim, out_features=self.emb_dim, bias=self.qkv_bias)
        self.W_value: nn.Linear = nn.Linear(in_features=self.emb_dim, out_features=self.emb_dim, bias=self.qkv_bias)

        self.register_buffer('mask', torch.triu(torch.ones((self.context_length, self.context_length)), diagonal=1), persistent=False)

        self.out_proj: nn.Linear = nn.Linear(in_features=self.emb_dim, out_features=self.emb_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, num_tokens, d_in = x.shape

        # [batch, num_tokens, emb_dim]
        queries: torch.Tensor = self.W_query(x)
        keys: torch.Tensor = self.W_key(x)
        values: torch.Tensor = self.W_value(x)

        # [batch, num_tokens, num_heads, head_dim]
        queries = queries.view(batch, num_tokens, self.num_heads, self.head_dim)
        keys = keys.view(batch, num_tokens, self.num_heads, self.head_dim)
        values = values.view(batch, num_tokens, self.num_heads, self.head_dim)

        # [batch, num_heads, num_tokens, head_dim]
        queries = queries.transpose(dim0=1, dim1=2)
        keys = keys.transpose(dim0=1, dim1=2)
        values = values.transpose(dim0=1, dim1=2)

        '''
        queries: [batch, num_heads, num_tokens, head_dim]
        keys.transpose: [batch, num_heads, head_dim, num_tokens]
        attention_scores: [batch, num_heads, num_tokens, num_tokens]
        '''
        attention_scores: torch.Tensor = queries @ keys.transpose(dim0=2, dim1=3)
        mask_bool: torch.Tensor = self.mask.bool()[: num_tokens, :num_tokens]
        attention_scores.masked_fill_(mask_bool, -torch.inf)

        # [batch, num_heads, num_tokens, num_tokens]
        attention_weights: torch.Tensor = torch.softmax(attention_scores / keys.shape[-1] ** 0.5, dim=-1)
        attention_weights = self.dropout(attention_weights)

        '''
        attention_weights: [batch, num_heads, num_tokens, num_tokens]
        values: [batch, num_heads, num_tokens, head_dim]
        context_vector: [batch, num_heads, num_tokens, head_dim] -> before transpose
        context_vector: [batch, num_tokens, num_heads, head_dim] -> after transpose

        '''
        context_vector: torch.Tensor = (attention_weights @ values).transpose(dim0=1, dim1=2)
        context_vector = context_vector.contiguous().view(batch, num_tokens, d_in)
        return self.out_proj(context_vector)
    

class TransformerBlock(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()

        self.norm1: LayerNorm = LayerNorm(cfg=cfg)
        self.multi_head_attention: MultiHeadAttention = MultiHeadAttention(cfg=cfg)
        self.dropout: nn.Dropout = nn.Dropout(p=cfg['drop_rate'])
        self.norm2: LayerNorm = LayerNorm(cfg=cfg)
        self.feed_forward: FeedForward = FeedForward(cfg=cfg)
    
    def forward(self, input_embeddings: torch.Tensor) -> torch.Tensor:
        skip_cell = input_embeddings
        x = self.norm1(input_embeddings)
        x = self.multi_head_attention(x)
        x = self.dropout(x)
        x = x + skip_cell

        skip_cell = x

        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = x + skip_cell
        return x
    

class GPTModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()

        self.token_embeddings: nn.Embedding = nn.Embedding(num_embeddings=cfg['vocab_size'], embedding_dim=cfg['emb_dim'])
        self.position_embeddings: nn.Embedding = nn.Embedding(num_embeddings=cfg['context_length'], embedding_dim=cfg['emb_dim'])

        self.dropout: nn.Dropout = nn.Dropout(p=cfg['drop_rate'])

        self.transformer_block: nn.Sequential = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )

        self.final_layer_norm: LayerNorm = LayerNorm(cfg=cfg)
        self.out_head: nn.Linear = nn.Linear(in_features=cfg['emb_dim'], out_features=cfg['vocab_size'], bias=False)

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        batch, seq_len = input_tokens.shape
        
        tok_emb: torch.Tensor = self.token_embeddings(input_tokens)
        pos_emb: torch.Tensor = self.position_embeddings(torch.arange(0, seq_len, device=input_tokens.device))

        x: torch.Tensor = tok_emb + pos_emb
        x = self.dropout(x)
        x = self.transformer_block(x)
        x = self.final_layer_norm(x)
        x = self.out_head(x)
        return x
    
class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        super().__init__()
        self.tokenizer: tiktoken = tokenizer
        self.tokens = self.tokenizer.encode(txt, allowed_special={'<|endoftext|>'})
        self.inputs: list = []
        self.targets: list = []

        for i in range(0, len(self.tokens) - max_length, stride):
            input_chunk = self.tokens[i: i + max_length]
            target_chunk = self.tokens[i+1: i + max_length + 1]

            self.inputs.append(torch.tensor(input_chunk))
            self.targets.append(torch.tensor(target_chunk))
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]
    

def create_data_loader(txt, cfg: dict):
    tokenizer = tiktoken.get_encoding('gpt2')
    dataset = GPTDataset(txt, tokenizer, max_length=cfg['context_length'], stride=cfg['context_length'] // 4)
    return DataLoader(dataset=dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=0, drop_last=True,)

def calculate_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss

def evaluate_model(model, train_loader, val_loader, device):
    model.eval()
    with torch.no_grad():
        train_loss = calculate_loss_loader(train_loader, model, device)
        val_loss = calculate_loss_loader(val_loader, model, device)
    model.train()
    return train_loss, val_loss

def calculate_loss_loader(data_loader, model, device, num_batches=5):
    total_loss = 0
    if len(data_loader) == 0:
        return float('nan')
    elif num_batches is None:
        num_batches = len(data_loader)

    else:
        num_batches = min(num_batches, len(data_loader))
    
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calculate_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    
    return total_loss / num_batches

def text_to_token_ids(txt, tokenizer):
    encoded = tokenizer.encode(txt, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)
        
        if temperature > 0.0:
            logits = logits / temperature

            probs = torch.softmax(logits, dim=-1) 

            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        
        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=-1)

    return idx
    

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs, start_context, tokenizer,):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    eval_freq = 5

    for epoch in range(num_epochs):
        model.train()

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calculate_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device,)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(start_context, tokenizer=tokenizer).to('cuda'),
            max_new_tokens=15,
            context_size=256,
            top_k=25,
            temperature=1.4,

        )

        print('Output Text: \n', token_ids_to_text(token_ids, tokenizer))
        
    return train_losses, val_losses, track_tokens_seen

def main(cfg: dict):
    filepath = './../ch02/data.txt'
    with open(file=filepath, mode='r', encoding='utf-8') as f:
        data = f.read()

    train_ratio = len(data) * 0.9
    train_data = data[:int(train_ratio)]
    val_data = data[int(train_ratio): ]

    train_dataset = create_data_loader(train_data, cfg=cfg)
    val_dataset = create_data_loader(val_data, cfg=cfg,)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPTModel(cfg=cfg)
    model.to(device=device)

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=1e-4, weight_decay=0.1)

    num_epochs = 100

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_dataset, val_dataset, optimizer, device, num_epochs,
    )


def download_gpt2():
    import os
    import requests
    file_name = "gpt2-small-124M.pth"
    
    # file_name = "gpt2-medium-355M.pth"
    # file_name = "gpt2-large-774M.pth"
    # file_name = "gpt2-xl-1558M.pth"

    url = f"https://huggingface.co/rasbt/gpt2-from-scratch-pytorch/resolve/main/{file_name}"

    if not os.path.exists(file_name):
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        with open(file_name, "wb") as f:
            f.write(response.content)
        print(f"Downloaded to {file_name}")

if __name__ == '__main__':
    download_gpt2()