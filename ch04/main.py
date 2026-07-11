import tiktoken
import torch
import gpt_model, gpt_config


model = gpt_model.GPTModel(gpt_config.GPT_CONFIG_124M)



print(f'Total Params: {sum(p.numel() for p in model.parameters())}')

print(model.eval())

