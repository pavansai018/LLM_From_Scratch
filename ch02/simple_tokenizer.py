import os
import re



class SimpleTokenizerV1:
    def __init__(self, file_path):
        self.raw_text = self.get_data(file_path)
        self.tokenized_data = self.tokenize(data=self.raw_text)
        self.vocab = self.create_vocabulary(self.tokenized_data)
        self.str_to_int = self.vocab
        self.int_to_str = {i: s for s, i in self.str_to_int.items()}
        self.pattern = r'([,.:;?_!"()\']|--|\s)'

    def encode(self, text):
        preprocessed = re.split(pattern=self.pattern, string=text)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        ids = [self.str_to_int[s] for s in preprocessed]
        return ids
    
    def decode(self, ids):
        text = ' '.join([self.int_to_str[i] for i in ids])
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)
        return text
    
    @staticmethod
    def get_data(file_path: str):
        with open(file=file_path, mode='r', encoding='utf-8') as f:
            raw_text = f.read()
        print(f'Total number of character: {len(raw_text)}')
        return raw_text
    
    @staticmethod
    def tokenize(data):
        pattern = r'([,.:;?_!"()\']|--|\s)'
        preprocessed = re.split(pattern=pattern, string=data)
        preprocessed = [item.strip() for item in preprocessed if item.strip()]
        preprocessed = [item if item in SimpleTokenizerV1.str_to_int else '<|unk|>' for item in preprocessed]
        print(f'Length of preprocessed: {len(preprocessed)}')
        return preprocessed

    @staticmethod
    def create_vocabulary(data):
        all_words = sorted(set(data))
        all_words.extend(['<|endoftext|>', '<|unk|>'])
        print(f'Vocab Size: {len(all_words)}')
        vocabulary = {token: integer for integer, token in enumerate(all_words)}
        return vocabulary

def main():
    tokenizer = SimpleTokenizerV1(file_path='./data.txt')
    text = 'Hello, the last he painted, you know,'
    ids = tokenizer.encode(text)
    print(ids)
    print(tokenizer.decode(ids))


if __name__ == '__main__':
    main()