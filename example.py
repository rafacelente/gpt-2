import torch
from torch.nn import Linear, GELU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout, functional as F
from typing import Optional, Tuple, List

MAX_CONTEXT = 128
dim = 768
n_heads = 12
embed_dim = 768
context = 50 # simulating the 50th word in the context
batch_size = 4

c_attn = Linear(dim, dim*3, bias=True) # W_q, W_k, W_v, that's why dim*3
c_proj = Linear(dim, dim, bias=True)

x = torch.randn(batch_size, context, embed_dim) # (batch_size, context, embed_dim)


def split_heads(x):
    return x.view(x.shape[0], x.shape[1], n_heads, dim//n_heads)

def merge_heads(x: torch.Tensor, num_heads, head_dim) -> torch.Tensor:
        return x.view((x.shape[0], x.shape[1], num_heads * head_dim))

def attention(q, k, v, mask=None):
    w = torch.matmul(q.transpose(1,2), k.transpose(1, 2).transpose(2, 3))
    w = w / torch.sqrt(torch.tensor(k.shape[-1]).float())
    if mask is not None:
        w = w + mask
    query_len = q.shape[1]
    key_len = k.shape[1]
    # Implementing the mask
    causal_mask = torch.tril(torch.ones((query_len, key_len), dtype=torch.bool))
    mask_value = torch.finfo(w.dtype).min # represent -inf
    w = torch.where(causal_mask, w, mask_value)
    
    w = F.softmax(w, dim=-1)
    attn_output = torch.matmul(w, v.transpose(1, 2)).transpose(1, 2)
    return attn_output

# Forward operation
xqkv = c_attn(x)
queries, keys, values = xqkv.split(dim, dim=2)
queries = split_heads(queries)
keys = split_heads(keys)
values = split_heads(values)

attn_output = attention(queries, keys, values)
attn_output = merge_heads(attn_output, n_heads, dim//n_heads)
attn_output = c_proj(attn_output)