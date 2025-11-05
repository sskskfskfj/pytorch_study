import torch
import torch.nn as nn
import torch.optim as optim
import torch.functional as F
import math


# Input Embedding
class InputEmbedding(nn.Module):
    def __init__(self, d_model, vocab_size):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len, dropout):
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        _2i = torch.exp(torch.arange(0, d_model, 2, dtype = torch.float))
        pe[:, 0::2] = torch.sin(position / 10000 ** (_2i / d_model))
        pe[:, 1::2] = torch.cos(position / 10000 ** (_2i / d_model))
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)

# Layer Normalization
class LayerNormalization(nn.Module):
    def __init__(self, eps = 1e-6):
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        std = x.std(dim = -1, keepdim = True)

        return self.alpha * (x - mean) / (std + self.eps) + self.bias

if __name__ == "__main__":
    positional_encoding = PositionalEncoding(d_model=512, seq_len=100, dropout=0.1)
    print(positional_encoding)