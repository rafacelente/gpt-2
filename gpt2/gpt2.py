import tiktoken
from gpt2.blocks import Transformer
from gpt2.modules import GPT2Module, TextDataModule
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Literal

class GPT2:
    @staticmethod
    def build(
            data: str, 
            model_size:str="gpt2", 
            input_type:Literal["parquet", "text", "path"]="parquet",
            max_length:int=1024,
            batch_size:int=8,
            train_test_split:float=0.8,
        ):
        tokenizer = tiktoken.get_encoding(model_size)
        # TODO: Change model based on model_size
        model = Transformer(
            dim=768,
            n_heads=12,
            vocab_size=50257,
            n_layers=12,
            max_seq_len=1024,
        )
        module = GPT2Module(model, tokenizer)

        if input_type == "path":
            with open(data, 'r') as f:
                data = f.read()

        datamodule = TextDataModule(
                        data, 
                        tokenizer, 
                        batch_size=batch_size, 
                        max_length=max_length, 
                        input_type=input_type,
                        train_test_split=train_test_split, 
                        seed=42)
        return GPT2(module, datamodule)

    
    def __init__(self, module: GPT2Module, datamodule: TextDataModule):
        self.module = module
        self.datamodule = datamodule
        self.trainer = None

    def train(self, max_epochs: int = 1, devices: int = 1, accelerator: str = "gpu"):
        self.trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
        )
        self.trainer.fit(self.module, self.datamodule)
        
    def generate(
            self,
            prompt,
            max_len=50,
            do_sample=True, 
            temperature=0.1, 
            top_k=0,
            top_p=0.9, 
            repetition_penalty=1.0, 
            num_return_sequences=1, 
            batch_size=1,
            device="cuda"):
        prompt_tokens = self.module.tokenizer.encode(prompt)
        self.module.model = self.module.model.to(device)
        for _ in range(num_return_sequences):
            generated = torch.tensor([prompt_tokens])
            prompt_len = len(prompt_tokens)
            generated = generated.to(device)

            for _ in range(max_len):
                with torch.no_grad():
                    outputs = self.module.model(generated)
                    next_token_logits = outputs[0][:, -1, :]
                    # for token in set(generated[0].tolist()):
                    #     next_token_logits[token] /= repetition_penalty
                    #next_token_logits = next_token_logits / temperature
                    #filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
                    if do_sample:
                        next_token = torch.multinomial(F.softmax(next_token_logits, dim=-1), num_samples=1)
                    # else:
                    #     next_token = torch.argmax(filtered_logits, dim=-1)
                    generated = torch.cat((generated, next_token), dim=1)

            result = generated[0].tolist()
            text = self.module.tokenizer.decode(result[prompt_len:])
        return text