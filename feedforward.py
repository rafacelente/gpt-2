import torch

class FeedForward(torch.nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.1, device="cuda"):
        super().__init__()
        self.c_fc = torch.nn.Linear(dim, hidden_dim, device=device)
        self.dropout = torch.nn.Dropout(dropout)
        self.c_proj = torch.nn.Linear(hidden_dim, dim, device=device)
    
    def forward(self, x):
        x = self.c_fc(x)
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.c_proj(x)
        return x