# import tiktoken
from src.blocks import Transformer
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import tiktoken

class GPT2Module(pl.LightningModule):
    def __init__(self, model:Transformer, tokenizer:object):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self.model(x, y)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self.model(x, y)
        self.log("val_loss", loss)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, _ = batch
        logits, _ = self.model(x)
        return logits

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)