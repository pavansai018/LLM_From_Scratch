from __future__ import annotations
from functools import partial
from typing import Any
import tiktoken
import torch
import torch.nn as nn
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as TorchDataset
import config


class ToxicDataset(TorchDataset):
    def __init__(self, dataset: TorchDataset, cfg: dict = config.GPT2_SMALL_124M):
        super().__init__()

        self.instruction: str = 'Rewrite the following toxic message in neutral and respectful language while preserving its non-harmful meaning.'
        self.context_length: int = cfg['context_length']
        self.eos_token_id: int = 50256
        self.tokenizer = tiktoken.get_encoding('gpt2')
        self.required_columns: set = {'en_toxic_comment', 'en_neutral_comment'}
        self.available_columns: set = set(dataset.column_names)
        
        if not self.required_columns.issubset(self.available_columns):
            raise ValueError(f'Expected Columns: {self.required_columns}\nAvailable Columns: {self.available_columns}')
        
        self.samples: list = []

        for row in dataset:
            toxic_text = str(row['en_toxic_comment']).strip()
            neutral_text = str(row['en_neutral_comment']).strip()

            if not toxic_text or not neutral_text:
                continue

            prompt = f'### Instruction:\n{self.instruction}\n\n### Input:\n{toxic_text.strip()}\n\n### Response:\n'
            prompt_ids = self.tokenizer.encode(prompt)
            response_ids = self.tokenizer.encode(neutral_text)

            complete_sequence = prompt_ids + response_ids + [self.eos_token_id]
            complete_sequence = complete_sequence[:self.context_length+1]

            if len(complete_sequence) <= len(prompt_ids):
                continue

            input_ids = complete_sequence[:-1]
            labels = complete_sequence[1:].copy()

            number_of_prompt_labels = len(prompt_ids) - 1
            labels[:number_of_prompt_labels] = [-100] * number_of_prompt_labels

            self.samples.append(
                {
                    'input_ids': torch.tensor(input_ids, dtype=torch.long),
                    'labels': torch.tensor(labels, dtype=torch.long),
                }
            )

            if len(self.samples) == 0:
                raise RuntimeError('No valid samples were created')

        print(f'Prepared {len(self.samples)} tokenized samples')


    def __len__(self,):
        return len(self.samples)
    

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]
    
class GPTBatchCollator:
    def __init__(self, pad_token_id: int = 50256, ignore_index: int = -100):
        self.pad_token_id = pad_token_id
        self.ignore_index = ignore_index

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        batch_size = len(batch)
        longest_sequence = max(sample['input_ids'].size(0) for sample in batch)
        input_ids = torch.full(size=(batch_size, longest_sequence), fill_value=self.pad_token_id, dtype=torch.long)
        labels = torch.full(size=(batch_size, longest_sequence), fill_value=self.ignore_index, dtype=torch.long)

        for batch_index, sample in enumerate(batch):
            sequence_length = sample['input_ids'].size(0)
            input_ids[batch_index, :sequence_length] = sample['input_ids']
            labels[batch_index, :sequence_length] = sample['labels']

        return {'input_ids': input_ids, 'labels': labels}


def create_dataloader():

    raw_dataset = load_dataset('s-nlp/paradetox', split='train')
    train_dataset = ToxicDataset(dataset=raw_dataset, cfg=config.GPT2_SMALL_124M)
    batch_collator = GPTBatchCollator()
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=batch_collator
    )

    # batch = next(iter(train_dataloader))

    # print("input_ids shape:", batch["input_ids"].shape,)
    # print("labels shape:", batch["labels"].shape,)

    # print("input_ids dtype:", batch["input_ids"].dtype,)
    # print("labels dtype:", batch["labels"].dtype,)
    return train_dataloader


if __name__ == '__main__':
    create_dataloader()