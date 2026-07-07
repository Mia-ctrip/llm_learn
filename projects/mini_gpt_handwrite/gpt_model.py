import torch
from torch import nn
import math



class mini_gpt(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, num_layers, max_length):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_length = max_length   
        self.embedding_layer = EmbeddingLayer(vocab_size, embed_size) 
        self.register_buffer('pos_enc', positional_encoding(embed_size, max_length))
        # 修复1：decoder_layer不需要传mask，hidden_size用4倍
        self.decoders = nn.ModuleList([decoder_layer(embed_size, num_heads, embed_size * 4) for _ in range(num_layers)])
        self.output_layer = output_layer(embed_size, vocab_size)

    def forward(self, x):
        x = self.embedding_layer(x)
        # 修复2：切片语法，pos_enc是3维(1, max_length, embed_size)
        x = self.pos_enc[:, :x.size(1), :] + x
        # 修复3：动态生成mask，传给每一层
        mask = random_mask(x.size(1)).to(x.device)
        for layer in self.decoders:
            x = layer(x, mask)
        x = self.output_layer(x)
        return x
    

class EmbeddingLayer(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embedding = nn.Embedding(vocab_size, embed_size)

    def forward(self, x):
        return self.embedding(x)



class MultiHeadAttention(nn.Module):
    # 修复4：去掉mask参数，mask在forward时传入
    def __init__(self, embed_size, num_heads):
        super().__init__()
        self.embed_size = embed_size
        self.num_heads = num_heads
        self.w_q = nn.Linear(embed_size, embed_size)
        self.w_o = nn.Linear(embed_size, embed_size)
        self.w_k = nn.Linear(embed_size, embed_size)
        self.w_v = nn.Linear(embed_size, embed_size)

    # 修复5：forward接收mask参数，从x读取batch和seq_len
    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)
        Q = Q.view(batch, seq_len, self.num_heads, self.embed_size // self.num_heads).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.embed_size // self.num_heads).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.embed_size // self.num_heads).transpose(1, 2)
        z = attention(Q, K, V, mask=mask)
        z = z.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_size)
        return self.w_o(z)

class FFN(nn.Module):
    def __init__(self, embed_size, hidden_size):
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.act = nn.GELU()
        self.fc1 = nn.Linear(embed_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, embed_size)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class decoder_layer(nn.Module):
    def __init__(self, embed_size, num_heads, hidden_size):
        super().__init__()
        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.ffn = FFN(embed_size, hidden_size)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

    # 修复6：接收mask，传给attention
    def forward(self, x, mask=None):
        # Pre-Norm
        x = x + self.attention(self.norm1(x), mask)
        x = x + self.ffn(self.norm2(x))
        return x

class output_layer(nn.Module):
    def __init__(self, embed_size, vocab_size):
        super().__init__()
        self.linear = nn.Linear(embed_size, vocab_size)

    def forward(self, x):
        return self.linear(x)

def positional_encoding(embed_size, max_length):
    pos = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-math.log(10000.0) / embed_size))
    pos_enc = torch.zeros(1, max_length, embed_size)
    pos_enc[:, :, 0::2] = torch.sin(pos * div_term)
    pos_enc[:, :, 1::2] = torch.cos(pos * div_term)
    return pos_enc

def attention(q, k, v, mask=None):
    attn = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(k.size(-1), dtype=torch.float))
    if mask is not None:
        attn = attn.masked_fill(mask == 0, float('-inf'))
    attn = torch.softmax(attn, dim=-1)
    return torch.matmul(attn, v)


def random_mask(seq_len):
    return torch.tril(torch.ones(seq_len, seq_len))
