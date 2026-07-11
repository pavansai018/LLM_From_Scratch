from config import GPT2_SMALL_124M, GPT2_MEDIUM_355M
import tiktoken
import torch
from main import GPTModel, generate, text_to_token_ids, token_ids_to_text

filename = '../models/gpt2_medium_model_and_optimizer.pth'


checkpoint = torch.load(filename, weights_only=True)

model = GPTModel(GPT2_MEDIUM_355M)
model.load_state_dict(checkpoint["model_state_dict"])

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
# model.train()
model.eval()

device = torch.device('cuda')
model.to(device)

torch.manual_seed(123)
tokenizer = tiktoken.get_encoding('gpt2')
text = 'Hi. How are you?'
token_ids = generate(
    model=model.to(device),
    idx=text_to_token_ids(text, tokenizer).to(device),
    max_new_tokens=30,
    context_size=GPT2_SMALL_124M['context_length'],
    top_k=1,
    temperature=1.0,
)

print('Output Text: \n', token_ids_to_text(token_ids, tokenizer))
