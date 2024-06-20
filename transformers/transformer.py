"""
implement transformer model from scratch
read the original paper: https://arxiv.org/abs/1706.03762
"""

import argparse
from typing import Any
import mlx.nn as nn
import mlx.core as mx

def custom_masked_fill(tensor: mx.array, mask: mx.array, fill_value):
    """
    Fills elements of `tensor` with `fill_value` where `mask` is True.
    
    Parameters:
    - tensor (mx.Array): The input tensor.
    - mask (mx.Array): A boolean tensor of the same shape as `tensor` indicating where to fill the value.
    - fill_value (float): The value to fill in `tensor` where `mask` is True.
    
    Returns:
    - Array.Array: The resulting tensor after applying the mask fill.
    """
    tensor[mask] = fill_value
    return tensor

class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(SelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.value = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.key = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.query = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(num_heads * self.head_dim, embed_dim)
    
    def __call__(self, values, keys, queries, mask):
        N = queries.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], queries.shape[1]
        # Split the embed_dim into num_heads
        values = values.reshape(N, value_len, self.num_heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.num_heads, self.head_dim)
        queries = queries.reshape(N, query_len, self.num_heads, self.head_dim)

        values = self.value(values)
        keys = self.key(keys)
        queries = self.query(queries)

        # Attention mechanism
        # energy = queries @ keys.T / sqrt(d_k)
        # transpose the last two dimensions of keys
        keys = mx.transpose(keys, axes=(0, 1, 3, 2))
        energy = queries @ keys
        energy = energy / self.head_dim ** 0.5
        if mask is not None:
            energy = custom_masked_fill(energy, mask, -1e10)
        # get the number of dimensions of the energy tensor
        attention = nn.softmax(energy, axis=-1)
        out = attention @ values
        out = out.reshape(N, query_len, self.embed_dim)
        out = self.fc_out(out)

        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout, forward_expansion):
        super(TransformerBlock, self).__init__()
        self.attention = SelfAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def __call__(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        # add skip connection and apply layer normalization
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out
    
def main():
    embed_dim = 512
    num_heads = 8
    forward_expansion = 4
    dropout = 0.1
    value_len = 10
    key_len = 10
    query_len = 10
    batch_size = 32

    value = mx.random.uniform(low=-0.1, high=0.1, shape=(batch_size, value_len, embed_dim))
    key = mx.random.uniform(low=-0.1, high=0.1, shape=(batch_size, key_len, embed_dim))
    query = mx.random.uniform(low=-0.1, high=0.1, shape=(batch_size, query_len, embed_dim))
    mask = mx.random.randint(0, 2, (batch_size, query_len, key_len))

    transformer_block = TransformerBlock(embed_dim, num_heads, dropout, forward_expansion)
    output = transformer_block(value, key, query, mask)
    print(output.shape)
    print(output)
    

if __name__ == "__main__":
    main()