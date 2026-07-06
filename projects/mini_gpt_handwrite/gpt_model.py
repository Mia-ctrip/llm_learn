import torch
from torch import nn



class mini_gpt(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, max_length):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_length = max_length    

    

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, x):
        return self.embedding(x)



class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.w_q = nn.Linear(embed_size, embed_size)
        self.w_o = nn.Linear(embed_size, embed_size)
        self.w_k = nn.Linear(embed_size, embed_size)
        self.w_v = nn.Linear(embed_size, embed_size)

    def forward(self):
        z =  attention(self.w_q(w_q), self.w_k(w_k), self.w_v(w_v), mask=True)
        return torch.matmul(z, self.w_o)

class FFN(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embed_size)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))

def positional_encoding(embed_size, max_length):
    pos = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-torch.log(torch.tensor(max_length)) / embed_size))
    pos_enc = torch.zeros(1, max_length, embed_size)
    pos_enc[:, :, 0::2] = torch.sin(pos * div_term)
    pos_enc[:, :, 1::2] = torch.cos(pos * div_term)
    return pos_enc

def attention(q, k, v, mask=None):
    attn = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float))
    if mask is not None:
        attn = attn.masked_fill(mask == 0, -1e9)
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)
