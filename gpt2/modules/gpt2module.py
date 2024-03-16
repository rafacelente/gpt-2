# import tiktoken
from gpt2.blocks import Transformer
import torch
import torch.nn.functional as F
import pytorch_lightning as pl

class GPT2Module(pl.LightningModule):
    def __init__(self, model:Transformer):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self.model(x, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        _, loss = self.model(x, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def predict_step(self, batch, batch_idx, dataloader_idx=None):
        x, _ = batch
        logits, _ = self.model(x)
        return logits

    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=1e-4, weight_decay=0.1)        