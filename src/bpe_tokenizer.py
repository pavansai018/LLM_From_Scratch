import tiktoken
import torch

class GPT2_Tokenizer:
    def __init__(self, tokenizer: tiktoken, encoding_name: str = 'gpt2'):
        self.tokenizer = tokenizer.get_encoding(encoding_name=encoding_name)
        self.vocab_size = self.tokenizer.n_vocab
        self.eos = '<|endoftext|>'
        self.eos_token_id = self.tokenizer.encode(self.eos, allowed_special={self.eos})[0]

    def encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(text=text, allowed_special={self.eos})

    def decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(tokens=token_ids)
    
    def encode_text_to_tensor(self, texts: list[str], add_eos_between_files: bool = True) -> torch.Tensor:
        all_tokens: list = []
        for idx, text in enumerate(texts):
            token_ids = self.encode(text=text)
            all_tokens.extend(token_ids)

            if add_eos_between_files:
                all_tokens.append(self.eos_token_id)

        return torch.tensor(all_tokens, dtype=torch.long) 
    