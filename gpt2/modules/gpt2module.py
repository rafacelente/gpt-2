from gpt2.blocks import Transformer
import torch
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
    
    def load_weights_from_hf(self, model_name: str="gpt2"):
        from transformers import GPT2Model
        hf_model = GPT2Model.from_pretrained(model_name)
        old_hf_state_dict = hf_model.state_dict()

        key_mapping = {
            'ln_1': 'norm1',
            'ln_2': 'norm2',
            'ln_f': 'norm_f',
        }

        hf_state_dict = {}
        for key, value in old_hf_state_dict.items():
            new_key = key
            for k, v in key_mapping.items():
                new_key = new_key.replace(k, v)
            if ".attn.bias" not in key and "masked_bias" not in key:    
                hf_state_dict[new_key] = value

        for key in list(hf_state_dict.keys()):
            if "c_attn.weight" in key or "mlp.c_fc.weight" in key or "mlp.c_proj.weight" in key:
                hf_state_dict[key] = hf_state_dict[key].T
        
        hf_state_dict["lm_head.weight"] = hf_state_dict["wte.weight"]
        
        self.model.load_state_dict(hf_state_dict)