from pytorch_lightning import LightningDataModule
from gpt2.modules.data.text_dataset import TextDataset
from torch.utils.data import DataLoader, default_collate, random_split
import torch
from typing import Optional
from torch.nn.utils.rnn import pad_sequence


def collate_batch(batch, max_length=1024):
    input_ids, labels = zip(*batch)

    max_len = max(len(x) for x in input_ids)
    padded_inputs = pad_sequence([torch.tensor(input_id) for input_id in input_ids], batch_first=True, padding_value=0)
    padded_labels = pad_sequence([torch.tensor(label) for label in labels], batch_first=True, padding_value=-100)
    
    if max_len > max_length:
        padded_inputs = padded_inputs[:, :1024]
        padded_labels = padded_labels[:, :1024]

    return padded_inputs, padded_labels

class TextDataModule(LightningDataModule):
    def __init__(self, data, tokenizer, batch_size:int=8, max_length:int=1024, input_type:Optional[str]="parquet", train_test_split:float=0.8, seed:int=42):
        super().__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_length = max_length
        self.input_type = input_type
        self.train_test_split = train_test_split
        self.seed = seed
        self.text_train = None
        self.text_val = None
        self.text_predict = None

    def setup(self, stage=None):
        if self.input_type == "parquet":
            dataset = TextDataset.from_parquet(self.data, self.tokenizer, max_length=self.max_length)
        elif self.input_type == "text":
            dataset = TextDataset.from_file(self.data, self.tokenizer, max_length=self.max_length)
        
        if stage == "fit":
            train_size = int(len(dataset) * self.train_test_split)
            val_size = len(dataset) - train_size
            self.text_train, self.text_val = random_split(
                dataset, 
                [train_size, val_size], 
                generator=torch.Generator().manual_seed(self.seed)
            )
        if stage == "predict":
            self.text_predict = dataset

    def train_dataloader(self):
        return DataLoader(
            self.text_train, 
            batch_size=self.batch_size, 
            shuffle=True, num_workers=0, 
            collate_fn=lambda x: collate_batch(x, max_length=self.max_length)
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.text_val, 
            batch_size=self.batch_size, 
            shuffle=False, num_workers=0, 
            collate_fn=lambda x: collate_batch(x, max_length=self.max_length)
        )
    
    def predict_dataloader(self):
        return DataLoader(
            self.text_predict,
            batch_size=self.batch_size, 
            shuffle=False, num_workers=0, 
            collate_fn=lambda x: collate_batch(x, max_length=self.max_length)
        )