import torch
from bpe_tokenizer import GPT2_Tokenizer
from embedding import InputEmbedding
import torch.nn as nn
import tiktoken
from config import GPT2_SMALL_124M
from attention import CausalMultiHeadAttention
from dataset import create_dataloader

class TransfromerBlock(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.multi_head_attention: CausalMultiHeadAttention = CausalMultiHeadAttention(
            d_in=cfg['emb_dim'],
            d_out=cfg['emb_dim'],
            context_length=cfg['context_length'],
            num_heads=cfg['n_heads'],
            dropout=cfg['drop_rate'],
            qkv_bias=cfg['qkv_bias'],
        )
        self.feedforward: FeedForward = FeedForward(cfg=cfg)
        self.norm1: LayerNorm = LayerNorm(cfg=cfg)
        self.norm2: LayerNorm = LayerNorm(cfg=cfg)
        self.drop_shortcut: torch.Tensor = nn.Dropout(p=cfg['drop_rate'])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut: torch.Tensor  = x
        x = self.norm1.forward(x)
        x = self.multi_head_attention.forward(x)
        x = self.drop_shortcut(x)
        x = x + shortcut # add the original input back

        # shortcut connection for feed forward block
        shortcut = x
        x = self.norm2.forward(x)
        x = self.feedforward.forward(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
    
    
class LayerNorm(nn.Module):
    def __init__(self, cfg: dict, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.shift: torch.Tensor = nn.Parameter(torch.zeros(cfg['emb_dim']))
        self.scale: torch.Tensor = nn.Parameter(torch.ones(cfg['emb_dim']))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean: torch.Tensor = x.mean(dim=-1, keepdim=True)
        var: torch.Tensor = x.var(dim=-1, keepdim=True, unbiased=False)
        norm_x: torch.Tensor = (x - mean) / (torch.sqrt(var + self.eps))
        norm_x: torch.Tensor = self.scale * norm_x + self.shift
        # print(norm_x.mean(dim=-1, keepdim=True))
        # print(norm_x.var(dim=-1, keepdim=True))
        return norm_x

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x) -> torch.Tensor:
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x, 3))))
    
class FeedForward(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.layers: torch.Tensor = nn.Sequential(
            nn.Linear(in_features=cfg['emb_dim'], out_features=4*cfg['emb_dim']),
            GELU(),
            nn.Linear(in_features=4*cfg['emb_dim'], out_features=cfg['emb_dim'])
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)
    
    

class GPTModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        self.token_embeddings: torch.Tensor = nn.Embedding(num_embeddings=cfg['vocab_size'], embedding_dim=cfg['emb_dim'])
        self.position_embeddings: torch.Tensor = nn.Embedding(num_embeddings=cfg['context_length'], embedding_dim=cfg['emb_dim'])
        self.dropout_for_embeddings: torch.Tensor = nn.Dropout(p=cfg['drop_rate'])

        self.transformer_blocks: torch.Tensor = nn.Sequential(
            *[TransfromerBlock(cfg) for _ in range(cfg['n_layers'])]
        )


        self.final_layer_norm: torch.Tensor = LayerNorm(cfg)
        self.out_head: torch.Tensor = nn.Linear(in_features=cfg['emb_dim'], out_features=cfg['vocab_size'], bias=False)

    def forward(self, input_tokens: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_tokens.shape
        token_embeddings = self.token_embeddings(input_tokens)
        position_embeddings = self.position_embeddings(torch.arange(seq_len, device=input_tokens.device))
        x = token_embeddings + position_embeddings
        x = self.dropout_for_embeddings(x)
        x = self.transformer_blocks(x)
        x = self.final_layer_norm(x)
        logits = self.out_head(x)
        return logits
    

def generate_text_simple(model, idx, max_new_tokens, context_size):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)


    return idx

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # New (not in book): numerical stability tip to get equivalent results on mps device
            # subtract rowwise max before softmax
            logits = logits - logits.max(dim=-1, keepdim=True).values
            
            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0) # add batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0) # remove batch dimension
    return tokenizer.decode(flat.tolist())

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # Main training loop
    for epoch in range(num_epochs):
        model.train()  # Set model to training mode
        
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward() # Calculate loss gradients
            optimizer.step() # Update model weights using loss gradients
            tokens_seen += input_batch.numel()
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # Print a sample text after each epoch
        # generate_and_print_sample(
        #     model, tokenizer, device, start_context
        # )
        token_ids = generate(
            model=model,
            idx=text_to_token_ids("Every effort moves you", tokenizer).to('cuda'),
            max_new_tokens=15,
            context_size=GPT2_SMALL_124M["context_length"],
            top_k=25,
            temperature=1.4
        )

        print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.position_embeddings.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))  # Compact print format
    model.train()

def main():
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding('gpt2')
    model = GPTModel(GPT2_SMALL_124M)

    with open('../ch02/data.txt', 'r') as f:
        data = f.read()

    train_ratio = 0.9
    split_idx = int(train_ratio * len(data))
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    # print(len(train_data), len(val_data))

    train_loader = create_dataloader(
        train_data,
        batch_size=2,
        max_length=GPT2_SMALL_124M['context_length'],
        stride=GPT2_SMALL_124M['context_length'],
        drop_last=True,
        shuffle=True,
        num_workers=0,
    )

    val_loader = create_dataloader(
        val_data,
        batch_size=2,
        max_length=GPT2_SMALL_124M['context_length'],
        stride=GPT2_SMALL_124M['context_length'],
        drop_last=False,
        shuffle=False,
        num_workers=0,
    )

    # for x, y in train_loader:
    #     print(x.shape, y.shape)

    # for x, y in val_loader:
    #     print(x.shape, y.shape)

    device = torch.device('cuda')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)

    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

if __name__ == '__main__':
    main()
    # tokenizer = tiktoken.get_encoding('gpt2')
    # batch = []
    # txt1 = 'Every effort moves you'
    # txt2 = 'Every day holds a'

    # batch.append(torch.tensor(tokenizer.encode(txt1)))
    # batch.append(torch.tensor(tokenizer.encode(txt2)))

    # batch = torch.stack(batch, dim=0)
    #tensor([[6109, 3626, 6100,  345],
    #        [6109, 1110, 6622,  257]])

    # print(batch)

    # torch.manual_seed(123)
    # model = GPTModel(cfg=GPT2_SMALL_124M)
    # print(batch.shape)
    # logits = model.forward(batch)
    # print(logits.shape)
    # print(logits)
    # total_params = sum(p.numel() for p in model.parameters())
 
    # print(f'{total_params - model.out_head.weight.numel():,}')
    # start_contetx = 'Hello, I am'
    # encoded = tokenizer.encode(start_contetx)
    # print('encoded: ', encoded)

    # encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    # print('encoded tensor.shape: ', encoded_tensor.shape)
    # out = generate_text_simple(model, encoded_tensor, 6, 1024)
    # print(out.shape)
    # print(out)
    # out = out.squeeze(0).tolist()
    # print(tokenizer.decode(out))
