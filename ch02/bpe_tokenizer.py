import tiktoken
import os

class BPETokenizer:
    def __init__(self):
        # self.raw_data = self.get_data(filepath=filepath)
        self.tokenizer = tiktoken.get_encoding('gpt2')


    @staticmethod
    def get_data(filepath: str):
        with open(file=filepath, mode='r', encoding='utf-8') as f:
            raw_data = f.read()
        return raw_data
    
    def _encode(self, text):
        return self.tokenizer.encode(text=text, allowed_special={'<|endoftext|>'})
    

d = BPETokenizer.get_data(filepath='./data.txt')
enc_text = BPETokenizer()._encode(d)
print(len(enc_text))