import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple, List

MAX_CONTEXT = 1024

class SelfAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1, device="cuda"):
        super().__init__()
        self.c_attn = nn.Linear(dim, dim*3, bias=True, device=device) # W_q, W_k, W_v, that's why dim*3
        self.c_proj = nn.Linear(dim, dim, bias=True, device=device)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout  = nn.Dropout(dropout)
        self.n_heads = n_heads
        self.dim = dim # Embedding dimension
        self.head_dim = dim // n_heads

        self.register_buffer(
            "bias", 
            torch.tril(torch.ones((MAX_CONTEXT, MAX_CONTEXT), dtype=torch.bool).view(1, 1, MAX_CONTEXT, MAX_CONTEXT)),
            persistent=False
        )

        # obs: we won't do cross attention here

    def _attn(self, q, k, v, mask=None):
        # q: (bsz, seqlen, n_heads, head_dim)
        # k: (bsz, seqlen, n_heads, head_dim)
        # v: (bsz, seqlen, n_heads, head_dim)
        # mask: (bsz, seqlen, seqlen)
        # return: (bsz, seqlen, n_heads, head_dim), (bsz, seqlen, seqlen)

        
        # We want to compute the attention weights for each token in the sequence
        # Therefore, the attention weights will be of shape (bsz, n_heads, seqlen, seqlen)
        # q.transpose(1,2): (bsz, n_heads, seqlen, head_dim)
        # k.transpose(1,2).transpose(2,3): (bsz, n_heads, head_dim, seqlen)
        w = torch.matmul(q.transpose(1,2), k.transpose(1,2).transpose(2,3))

        # We will scale the weights by default as in the paper,
        # but this may be a configurable parameter
        w = w / torch.sqrt(torch.tensor(k.shape[-1]).float())

        query_len = q.shape[1]
        key_len = k.shape[1]
        # Implementing the mask
        #causal_mask = torch.tril(torch.ones((query_len, key_len), dtype=torch.bool, device=q.device))
        causal_mask = self.bias[:, :, key_len-query_len : key_len, :key_len]
        mask_value = torch.finfo(w.dtype).min # represent -inf
        w = torch.where(causal_mask, w, mask_value)

        w = self.attn_dropout(F.softmax(w, dim=-1))

        attn_output = torch.matmul(w, v.transpose(1, 2)).transpose(1, 2)
        return attn_output

    def _merge_heads(self, x: torch.Tensor, num_heads, head_dim) -> torch.Tensor:
        # x: (bsz, seqlen, n_heads, head_dim)
        # return: (bsz, seqlen, dim)
        return x.contiguous().view(x.shape[0], x.shape[1], num_heads * head_dim)
    
    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (bsz, seqlen, dim)
        # return: (bsz, seqlen, n_heads, head_dim)
        return x.view(x.shape[0], x.shape[1], self.n_heads, self.head_dim)

    # def _update_cache(cache_kv, keys, values, kv_len):
    #     # cache_kv: (2, bsz, MAX_CONTEXT, n_heads, head_dim)
    #     # keys: (bsz, kv_len, n_heads, head_dim)
    #     # values: (bsz, kv_len, n_heads, head_dim)
    #     # kv_len: length of keys and values
    #     # return: (2, bsz, MAX_CONTEXT, n_heads, head_dim)
    #     cache_k, cache_v = cache_kv[0], cache_kv[1]
    #     cache_k = torch.cat([cache_k[:, :, kv_len:], keys], dim=2)
    #     cache_v = torch.cat([cache_v[:, :, kv_len:], values], dim=2)

    #     if cache_k.shape[2] > MAX_CONTEXT:
    #         cache_k = cache_k[:, :, -MAX_CONTEXT:]
    #         cache_v = cache_v[:, :, -MAX_CONTEXT:]

    #     return torch.stack([cache_k, cache_v])

    def forward(self, x: torch.Tensor, layer_past: Optional[Tuple[torch.Tensor]]=None, mask: Optional[None]=None):
        xqkv : torch.Tensor = self.c_attn(x) # (bsz, seqlen, dim*3)
        queries, keys, values = xqkv.split(self.dim, dim=2) # (bsz, seqlen, dim)
        queries = self._split_heads(queries) # (bsz, seqlen, n_heads, head_dim)
        keys = self._split_heads(keys) # (bsz, seqlen, n_heads, head_dim)
        values = self._split_heads(values) # (bsz, seqlen, n_heads, head_dim)
        
        bsz, seqlen = queries.shape[:2] # batch size, sequence length

        # if not hasattr(self, 'cache_kv'):
        #     self.cache_kv = torch.zeros((2, bsz, MAX_CONTEXT, self.n_heads, self.head_dim), dtype=x.dtype, device=x.device)
        
        # self.cache_kv = self._update_cache(self.cache_kv, key, value, seqlen)

        if layer_past is not None:
            past_keys, past_values = layer_past # this is essentially the cache_kv with no max_context
            keys = torch.cat([past_keys, keys], dim=2)
            values = torch.cat([past_values, values], dim=2)
        
        attn_output = self._attn(queries, keys, values, mask)
        attn_output = self._merge_heads(attn_output, self.n_heads, self.head_dim)
        attn_output = self.c_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)

        return attn_output
