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
            from_checkpoint: Optional[str] = None
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
        if from_checkpoint is not None:
            module = GPT2Module.load_from_checkpoint(from_checkpoint, model=model)
        elif from_pretrained:
            module = GPT2Module(model)
            module.load_weights_from_hf(model_size)
        else:
            module = GPT2Module(model)
        return GPT2(module, None, tokenizer=tokenizer)
    
    def __init__(self, module: GPT2Module, datamodule: TextDataModule, tokenizer: Optional[object] = None):
        self.module = module
        self.datamodule = datamodule
        self.trainer = None
        self.tokenizer = tokenizer

    def load_checkpoint(self, path: str):
        self.module = GPT2Module.load_from_checkpoint(path, model=self.module.model)

    def load_weights_from_hf(self, model_name: str = "gpt2"):
        assert self.module is not None, "module must be loaded before loading weights"
        self.module.load_weights_from_hf(model_name)

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
        prompt_tokens = self.tokenizer.encode(prompt)
        self.module.model = self.module.model.to(device)
        self.module.model.eval()

        def top_k_filtering(logits, top_k=0, filter_value=-float('Inf')):
            top_k = min(top_k, logits.size(-1))
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = filter_value

            return logits

        for _ in range(num_return_sequences):
            generated = torch.tensor([prompt_tokens])
            generated = generated.to(device)

            for _ in range(max_len):
                with torch.no_grad():
                    outputs = self.module.model(generated)
                    next_token_logits = outputs[0][:, -1, :]
                    for token in set(generated[0].tolist()):
                        next_token_logits[:, token] /= repetition_penalty
                    next_token_logits = next_token_logits / temperature
                    filtered_logits = top_k_filtering(next_token_logits, top_k=top_k)
                    if do_sample:
                        next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
                    else:
                        next_token = torch.argmax(F.softmax(filtered_logits, dim=-1), dim=-1, keepdims=True)
                    generated = torch.cat((generated, next_token), dim=-1)

            result = generated[0].tolist()
            text = self.tokenizer.decode(result)
        return text
    
    def load_datamodule(
            self,
            data_path: Union[str, List[str]],
            dataset: Optional[Literal["shakespeare", "tinystrange", "wikitext", "gepeto"]] = "gepeto",
            batch_size: Optional[int] = 8,
            train_test_split: Optional[float] = 0.8,
            max_length: Optional[int] = 1024,
        ):
        if dataset == "shakespeare":
            dataset = ShakespeareDataset.from_file(data_path, self.tokenizer, max_length)
        elif dataset == "tinystrange":
            dataset = TinyStrangeDataset.from_parquet(data_path, self.tokenizer, max_length)
        elif dataset == "wikitext":
            dataset = WikiTextDataset.from_parquet(data_path, self.tokenizer, max_length)
        elif dataset == "gepeto":
            dataset = GepetoDataset(data_path, self.tokenizer, max_length)
        else:
            raise ValueError(f"Invalid dataset {dataset}")
        
        self.datamodule = TextDataModule(
                        dataset, 
                        batch_size=batch_size, 
                        train_test_split=train_test_split, 
                        seed=42)
