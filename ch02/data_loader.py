import torch
from torch.utils.data import DataLoader, Dataset
import tiktoken

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        super().__init__()

        self.input_ids = []
        self.target_ids = []

        self.tokenizer = tokenizer
        token_ids = self.tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i: i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]
    

def create_dataloader_v1(txt, batch_size=4, max_length=256, stride=128, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding(encoding_name='gpt2')
    dataset = GPTDatasetV1(txt=txt, tokenizer=tokenizer, max_length=max_length, stride=stride)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=drop_last,
    )

    return dataloader

def main():
    filepath = r'./data.txt'
    with open(file=filepath, mode='r', encoding='utf-8') as f:
        data = f.read()
    
    dataloader = create_dataloader_v1(txt=data, batch_size=1, max_length=4, stride=1, shuffle=False)
    data_iter = iter(dataloader)
    first_batch = next(data_iter)
    print(first_batch)


if __name__ == '__main__':
    main()