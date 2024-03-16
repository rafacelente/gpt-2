import tiktoken
from gpt2.blocks import Transformer
from gpt2.modules import GPT2Module, TextDataModule
from gpt2.modules.data import ShakespeareDataset, TinyStrangeDataset, WikiTextDataset, GepetoDataset
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Literal, Optional, List, Union
class GPT2:
    @staticmethod
    def build(
            model_size: Optional[str] = "gpt2",
            max_length: Optional[int] = 1024,
            from_pretrained: Optional[bool] = False,
            checkpoint_path: Optional[str] = None
        ):
        tokenizer = tiktoken.get_encoding(model_size)
        # TODO: Change model based on model_size
        model = Transformer(
            dim=768,
            n_heads=12,
            vocab_size=50257,
            n_layers=12,
            max_seq_len=max_length,
        )
        if from_pretrained:
            assert checkpoint_path is not None, "checkpoint_path must be provided if from_pretrained is True"
            module = GPT2Module.load_from_checkpoint(checkpoint_path, model=model, tokenizer=tokenizer)
        else:
            module = GPT2Module(model, tokenizer)
        return GPT2(module, None)
    
    def load_checkpoint(self, path: str):
        self.module = GPT2Module.load_from_checkpoint(path, model=self.module.model, tokenizer=self.module.tokenizer)

    def __init__(self, module: GPT2Module, datamodule: TextDataModule):
        self.module = module
        self.datamodule = datamodule
        self.trainer = None

    def train(
            self, 
            max_epochs: int = 1, 
            devices: int = 1, 
            accelerator: str = "gpu", 
            logger: Optional[pl.loggers.Logger]=None,
            ):
        assert self.datamodule is not None, "datamodule must be loaded before training"
        self.trainer = pl.Trainer(
            accelerator=accelerator,
            devices=devices,
            max_epochs=max_epochs,
            logger=logger,
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
    
    def load_datamodule(
            self,
            dataset: Literal["shakespeare", "tinystrange", "wikitext", "gepeto"], # TODO: Add more datasets,
            data_path: Union[str, List[str]],
            tokenizer: object,
            batch_size: Optional[int] = 8,
            train_test_split: Optional[float] = 0.8,
            max_length: Optional[int] = 1024,
        ):
        if dataset == "shakespeare":
            dataset = ShakespeareDataset.from_file(data_path, tokenizer, max_length)
        elif dataset == "tinystrange":
            dataset = TinyStrangeDataset.from_parquet(data_path, tokenizer, max_length)
        elif dataset == "wikitext":
            dataset = WikiTextDataset.from_parquet(data_path, tokenizer, max_length)
        elif dataset == "gepeto":
            dataset = GepetoDataset(data_path, tokenizer, max_length)
        else:
            raise ValueError(f"Invalid dataset {dataset}")
        
        self.datamodule = TextDataModule(
                        dataset, 
                        batch_size=batch_size, 
                        train_test_split=train_test_split, 
                        seed=42)
