"""
implementation of Bahdanau attention mechanism following the paper "Neural Machine Translation by Jointly Learning to Align and Translate" by Bahdanau et al.
read the original paper here: https://arxiv.org/abs/1409.0473
"""

import mlx.nn as nn
import mlx.core as mx

class BahdanauAttention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super(BahdanauAttention, self).__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        # linear layer to transform the encoder hidden states
        self.attn = nn.Linear(enc_hid_dim + dec_hid_dim, dec_hid_dim)
        # linear layer to calculate the attention score
        self.v = nn.Linear(dec_hid_dim, 1, bias=False)
    
    def __call__(self, hidden, encoder_outputs):
        src_len = encoder_outputs.shape[1]
        hidden = mx.expand_dims(hidden, axis=1)
        # repeat the hidden state to match the shape of the encoder outputs
        hidden = mx.repeat(hidden, src_len, axis=1)
        # calculate the attention energy by applying the linear layer and tanh activation
        energy = mx.tanh(self.attn(mx.concatenate((hidden, encoder_outputs), axis=2)))
        # calculate the attention scores by applying the linear layer
        attention = self.v(energy).squeeze(2)
        return mx.softmax(attention, axis=0)

if __name__ == "__main__":
    attention = BahdanauAttention(256, 512)
    hidden = mx.random.uniform(low=-0.1, high=0.1, shape=(32, 512))
    encoder_outputs = mx.random.uniform(low=-0.1, high=0.1, shape=(32, 10, 256))
    output = attention(hidden, encoder_outputs)
    print(output.shape)
    print(output)