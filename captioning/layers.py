# The four building blocks used by the captioner

# Source credits are recorded in the project notice
# Shapes: N = images in a batch, T = words, D = embedding size

import math

import torch
from torch import nn


class PositionalEncoding(nn.Module):
    # Add a fixed sine/cosine pattern so word order is visible to attention

    def __init__(self, embed_dim, dropout=0.1, max_len=5000):
        super().__init__()
        if embed_dim <= 0 or embed_dim % 2:
            raise ValueError("embed_dim must be positive and even")
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(1, max_len, embed_dim)
        positions = torch.arange(max_len, dtype=torch.float32)
        # Fill one sine and cosine column pair at a time
        for column in range(0, embed_dim, 2):
            angles = positions / (10000 ** (column / embed_dim))
            pe[0, :, column] = torch.sin(angles)
            pe[0, :, column + 1] = torch.cos(angles)
        self.register_buffer("pe", pe)

    def forward(self, x):
        if x.shape[1] > self.pe.shape[1]:
            raise ValueError("Caption is longer than max_length")
        return self.dropout(x + self.pe[:, :x.shape[1]])


class MultiHeadAttention(nn.Module):
    # Compare queries to keys, then average values using those similarities

    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        if num_heads <= 0 or embed_dim % num_heads:
            raise ValueError("embed_dim must be divisible by num_heads")
        # Keep initialization order stable for the numerical checks
        self.key = nn.Linear(embed_dim, embed_dim)
        self.query = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.n_head = num_heads
        self.head_dim = embed_dim // num_heads

    def forward(self, query, key, value, attn_mask=None):
        # Inputs are (N, words, D); a True mask entry allows attention
        queries = self.query(query)
        keys = self.key(key)
        values = self.value(value)
        heads = []
        # Process each head separately
        # Each head reads its own slice of features
        for head in range(self.n_head):
            start = head * self.head_dim
            end = start + self.head_dim
            q = queries[:, :, start:end]
            k = keys[:, :, start:end]
            v = values[:, :, start:end]
            scores = q @ k.transpose(1, 2) / math.sqrt(self.head_dim)
            if attn_mask is not None:
                scores = scores.masked_fill(attn_mask == 0, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            weights = self.attn_drop(weights)
            heads.append(weights @ v)
        return self.proj(torch.cat(heads, dim=-1))


class FeedForwardNetwork(nn.Module):
    # Process each word independently after it has gathered context

    def __init__(self, embed_dim, ffn_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.dropout(x)
        return self.fc2(x)


class TransformerDecoderLayer(nn.Module):
    # Read earlier words, read the image, then refine each word's features

    def __init__(self, input_dim, num_heads, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(input_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(input_dim, dim_feedforward, dropout)
        self.norm_self = nn.LayerNorm(input_dim)
        self.norm_cross = nn.LayerNorm(input_dim)
        self.norm_ffn = nn.LayerNorm(input_dim)
        self.dropout_self = nn.Dropout(dropout)
        self.dropout_cross = nn.Dropout(dropout)
        self.dropout_ffn = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None):
        context = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm_self(tgt + self.dropout_self(context))
        image_context = self.cross_attn(tgt, memory, memory)
        tgt = self.norm_cross(tgt + self.dropout_cross(image_context))
        refined = self.ffn(tgt)
        return self.norm_ffn(tgt + self.dropout_ffn(refined))
