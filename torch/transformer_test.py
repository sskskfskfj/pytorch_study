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

# Feed Forward
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear_2(self.dropout(F.relu(self.linear_1(x))))

# Multi-Head Attention
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, head, dropout = 0.1):
        super().__init__()
        self.d_model = d_model
        self.head = head
        assert d_model % head == 0, "d_model must be divisible by head"
        
        self.d_k = d_model // head
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
    
    @staticmethod
    def attention(self, query, key, value, mask, dropout = None):
        d_k = query.shape[-1] # 왜?
        attention_score = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)
        attention_score = F.softmax(attention_score, dim = -1)

        if dropout is not None:
            attention_score = dropout(attention_score)

        return (attention_score @ value), attention_score

    def forward(self, query, key, value, mask = None):
        query = self.q(query)        
        key = self.k(key)
        value = self.v(value)

        query = query.view(query.size(0), query.size(1), self.head, self.d_k).transpose(1, 2)
        key = key.view(key.size(0), key.size(1), self.head, self.d_k).transpose(1, 2)
        value = value.view(value.size(0), value.size(1), self.head, self.d_k).transpose(1, 2)

        x, self.attention_score = self.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1,2).contiguous().view(x.shape[0],-1, self.h*self.d_k)  

        return self.o(x)
        
if __name__ == "__main__":
    positional_encoding = PositionalEncoding(d_model=512, seq_len=100, dropout=0.1)
    print(positional_encoding)