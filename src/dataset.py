from torch.utils.data import Dataset, DataLoader
import torch
from bpe_tokenizer import GPT2_Tokenizer

class GPTTokenDataset(Dataset):
    def __init__(self, token_ids: torch.Tensor, context_length: int):
        super().__init__()
        self.token_ids = token_ids
        self.context_length = context_length

    def __len__(self) -> int:
        return len(self.token_ids) - self.context_length
    
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.token_ids[index: index + self.context_length + 1]

        inputs = chunk[:-1]
        target = chunk[1:]

        return inputs, target
    

def create_gpt_dataloader(folder_path: str, batch_size: int = 8, context_length: int = 1024, shuffle: bool = True, num_workers: int = 0):
    text_reader = None
    texts = None

    tokenizer = GPT2_Tokenizer()
    token_ids = tokenizer.encode_text_to_tensor(texts=texts)

    dataset = GPTTokenDataset(
        token_ids=token_ids,
        context_length=context_length
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )

    return dataloader, tokenizer