import tiktoken
from transformer import Transformer
import torch
import torch.nn.functional as F

class GPT2:
    @staticmethod
    def build(model_size="gpt2"):
        tokenizer = tiktoken.get_encoding(model_size)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        model = Transformer(
            dim=768,
            n_heads=12,
            vocab_size=50257,
            n_layers=12,
            max_seq_len=1024,
            device=device
        )

        return GPT2(model, tokenizer)
    
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

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
        
        device = torch.device(device) if torch.cuda.is_available() else torch.device('cpu')
        prompt_tokens = self.tokenizer.encode(prompt)

        for _ in range(num_return_sequences):
            generated = torch.tensor([prompt_tokens])
            generated = generated.to(device)
            prompt_len = len(prompt_tokens)

            for _ in range(max_len):
                outputs = self.model(generated)
                print(outputs[1])
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
            text = self.tokenizer.decode(result[prompt_len:])
            return text