import torch
from torch import nn
from gpt2.blocks import SelfAttention, FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, norm_eps=1e-5, device="cuda"):
        super().__init__()
        self.attn = SelfAttention(dim, n_heads, device=device)
        self.norm1 = nn.LayerNorm(dim, eps=norm_eps, device=device)
        self.mlp = FeedForward(dim, dim*4, device=device)
        self.norm2 = nn.LayerNorm(dim, eps=norm_eps, device=device)

    def forward(self, x):
        residual = x
        attn_outputs = self.attn(self.norm1(x))
        x = residual + attn_outputs
        residual = x
        x = self.norm2(x)
        return self.mlp(x) + residual

class Transformer(nn.Module):
    def __init__(
            self, 
            dim, 
            n_heads,
            norm_eps=1e-5, 
            vocab_size=50257, 
            n_layers=12, 
            max_seq_len=1024, 
            device="cuda"):
        super().__init__()
        self.embed_dim = dim
        self.vocab_size = vocab_size
        self.wte = nn.Embedding(vocab_size, dim, device=device)
        self.wpe = nn.Embedding(max_seq_len, dim, device=device)
        self.h = nn.ModuleList([TransformerBlock(dim, n_heads, norm_eps, device=device) for _ in range(n_layers)])
        self.norm_f = nn.LayerNorm(dim, eps=norm_eps, device=device)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False, device=device)


    def forward(self, x, labels=None):
        input_shape = x.shape
        bsz, seq_len = input_shape[0], input_shape[1]
        
        positions = torch.arange(seq_len, dtype=torch.long, device=x.device)
        position_embeddings = self.wpe(positions)
        x = self.wte(x) + position_embeddings
        x = self.norm_f(x)
        for i in range(len(self.h)):
            x = self.h[i](x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.vocab_size), labels.view(-1))
        
        output = (logits,) + (loss,) if loss is not None else (logits,)

        return output