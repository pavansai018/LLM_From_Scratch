import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import json
from functools import partial
import tiktoken
from src import config
from src.main import GPTModel, generate, train_model_simple, text_to_token_ids, token_ids_to_text


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []

        # pre tokenize text
        for entry in self.data:
            instruction_plus_input = format_input(entry)
            response_text = f'\n\n### Response:\n{entry["output"]}'
            full_text = instruction_plus_input + response_text
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.encoded_texts[index]

def custom_collate(batch, pad_token_id=50256, ignore_index=-100, allowed_max_length=None, device='cpu'):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_list, targets_list = [], []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id] 
        padded = (new_item + [pad_token_id] * (batch_max_length - len(new_item)))
        inputs = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])

        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        if allowed_max_length is not None:
            inputs = inputs[: allowed_max_length]
            targets = targets[: allowed_max_length]

        inputs_list.append(inputs)
        targets_list.append(targets)

    inputs_tensor = torch.stack(inputs_list).to(device)
    targets_tensor = torch.stack(targets_list).to(device)

    return inputs_tensor, targets_tensor


def load_json(filepath):
    with open(file=filepath, mode='r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def format_input(entry):
    instruction_text = (
        f'Below is an instruction that describes a task. '
        f'Write a response that appropriately completes the request.'
        f'\n\n### Instruction:\n{entry["instruction"]}'
    )

    input_text = f'\n\n### Input:\n{entry["input"]}' if entry['input'] else ''

    return instruction_text + input_text


def main():
    torch.manual_seed(123)
    tokenizer = tiktoken.get_encoding('gpt2')
    filepath = 'supervised_instruction_finetuning/instruction_data.json'
    data = load_json(filepath)
    # print(format_input(data[50]))

    train_ratio = 0.85
    val_ratio = 0.1
    # test_ratio = 0.05
    total_data_points = len(data)
    train_data = data[:int(total_data_points * train_ratio)]
    val_data = data[int(total_data_points * train_ratio): int(total_data_points * train_ratio) + int(total_data_points * val_ratio)]
    test_data = data[len(train_data) + len(val_data):]
    # print(len(data), len(train_data), len(val_data), len(test_data))
    customized_collate_fn = partial(custom_collate, device='cpu', allowed_max_length=1024)
    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )

    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    test_dataset = InstructionDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    model = GPTModel(config.GPT2_MEDIUM_355M)
    model_weights = 'models/gpt2_medium_model_and_optimizer.pth'
    checkpoint = torch.load(model_weights, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0004, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.to('cuda')
    num_epochs = 10
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, 'cuda',
        num_epochs=num_epochs,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    print('*'*100)

    for entry in test_data[:3]:

        input_text = format_input(entry)

        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to('cuda'),
            max_new_tokens=256,
            context_size=config.GPT2_MEDIUM_355M["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = (
            generated_text[len(input_text):]
            .replace("### Response:", "")
            .strip()
    )

        print(input_text)
        print(f"\nCorrect response:\n>> {entry['output']}")
        print(f"\nModel response:\n>> {response_text.strip()}")
        print("-------------------------------------")

if __name__ == '__main__':
    main()