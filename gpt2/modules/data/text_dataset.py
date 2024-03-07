from torch.utils.data import Dataset
import torch
from typing import Optional

class ShakespeareDataset(Dataset):
    def __init__(self, entry: str, tokenizer: object, max_length: Optional[str]=1024):
        self.entry = entry
        self.inputs = []
        self.labels = []
        self.max_length = max_length
        self.tokenizer = tokenizer

    @classmethod
    def from_file(cls, file_path: str, tokenizer: object, max_length: Optional[int] =1024):
        with open(file_path, "r") as f:
            entry = f.read()
        dataset = cls(entry, tokenizer, max_length)
        dataset._prepare_text()
        return dataset
    
    def _prepare_text(self):
        tokenized_text = self.tokenizer.encode(self.entry)

        for i in range(0, len(tokenized_text), self.max_length):
            a = tokenized_text[i:i+self.max_length]
            b = tokenized_text[i+1:i+self.max_length+1]
            if len(a) != self.max_length:
                print(f"{i} this has a different length: {len(a)}, padding") 
                a = a + [0 for _ in range(self.max_length - len(a))]
                b = b + [0 for _ in range(self.max_length - len(b))]
            self.inputs.append(a)
            self.labels.append(b)

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.labels[idx])
    



class TinyStrangeDataset(Dataset):
    def __init__(self, file_path:str,  tokenizer: object, max_length: Optional[str]=1024):
        self.df = None
        self.file_path = file_path
        self.max_length = max_length
        self.tokenizer = tokenizer
    
    @classmethod
    def from_parquet(cls, file_path: str, tokenizer: object, max_length: int=1024, tokenized: bool=False):
        import pandas as pd
        dataset = cls(file_path, tokenizer, max_length=max_length)
        print(f'Reading parquet file {file_path}...')
        df = pd.read_parquet(file_path)
        if not tokenized:
            print('Tokenizing text...')
            dataset.df["tokens"] = dataset.df["text"].apply(lambda x: tokenizer.encode(x, allowed_special={'<|endoftext|>'}))
            dataset.df["len_tokens"] = dataset.df["tokens"].apply(lambda x: len(x))
        
        assert "tokens" in dataset.df.columns, "Column 'tokens' not found in dataframe"
        dataset.df = df
        return dataset
    

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        tokens = self.df.iloc[idx]["tokens"]
        tokens = tokens[:self.max_length]
        inputs = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)

        return inputs, labels