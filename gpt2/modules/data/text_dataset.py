from torch.utils.data import Dataset
import torch
import numpy as np


class TextDataset(Dataset):
    def __init__(self, entry, tokenizer, max_length=128, input_type="text"):
        self.entry = entry
        self.inputs = []
        self.labels = []
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.input_type = input_type

        self.prepare()

    def prepare(self):
        if self.input_type == "text":
            self._prepare_text()
        elif self.input_type == "bin":
            self._prepare_bin()
    
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
            
    def _prepare_bin(self):
        tokenized_bin = torch.from_numpy(np.array(self.entry)).long()

        for i in range(0, len(tokenized_bin), self.max_length):
            self.inputs.append(tokenized_bin[i:i+self.max_length])
            self.labels.append(tokenized_bin[i+1:i+self.max_length+1])

    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return torch.tensor(self.inputs[idx]), torch.tensor(self.labels[idx])