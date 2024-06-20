"""
implement bert model from scratch using transformer
read the original paper: https://arxiv.org/abs/1810.04805
"""

import argparse
from functools import partial
import time
from typing import Any
import mlx.nn as nn
import mlx.core as mx
import mlx.optimizers as optim
from mlx.utils import tree_flatten
import numpy as np
from transformers.transformer import TransformerBlock
from transformers.data import dummy_data, iterate_batches, to_samples


class BERT(nn.Module):
    """
    implementation of BERT model
    """
    def __init__(self, embed_dim, num_heads, dropout, forward_expansion, vocab_size, max_length):
        super(BERT, self).__init__()
        self.encoder = TransformerBlock(embed_dim, num_heads, dropout, forward_expansion)
        self.decoder = nn.Linear(embed_dim, vocab_size)
        self.word_embeddings = nn.Embedding(vocab_size, embed_dim)
        self.positional_embeddings = nn.Embedding(max_length, embed_dim)

    def __call__(self, x):
        positions = mx.expand_dims(mx.arange(x.shape[1]), axis=0)
        out = self.word_embeddings(x) + self.positional_embeddings(positions)
        mask = mx.random.randint(0, 2, shape=(x.shape[0], x.shape[1]))
        out = self.encoder(out, out, out, mask)
        out = self.decoder(out)
        return out


def main(args):
    # create the toy data to train the model with masked language modeling
    context_size = 10
    vocab, data_set = dummy_data()
    # Initialize the model
    embed_dim = 256
    num_heads = 2
    forward_expansion = 4
    batch_size = 4
    vocab_size = len(vocab)
    max_length = context_size
    dropout = 0.1
    # Training parameters
    lr_warmup = 200
    learning_rate = 3e-4
    weight_decay = 1e-5

    model = BERT(embed_dim, num_heads, dropout, forward_expansion, vocab_size, max_length)
    mx.eval(model.parameters())
    n_params = sum(
        x.size for k, x in tree_flatten(model.parameters()) if "embedding" not in k
    )
    print(f"Training a transformer with {n_params / 1024**2:.3f} M parameters")

    def loss_fn(model, x, y, reduce=True):
        logits = model(x)
        losses = nn.losses.cross_entropy(logits, y)
        return mx.mean(losses) if reduce else mx.mean(losses, axis=(-1, -2))

    optimizer = optim.AdamW(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    state = [model.state, optimizer.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def step(inputs, targets):
        loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
        loss, grads = loss_and_grad_fn(model, inputs, targets)
        optimizer.update(model, grads)
        return loss
    
    iterator = iterate_batches(batch_size, context_size, data_set)
    losses = []

    tic = time.perf_counter()
    for epoch, (inputs, targets) in zip(range(args.num_epochs), iterator):
        inputs, targets = map(mx.array, (inputs, targets))
        optimizer.learning_rate = min(1, epoch / lr_warmup) * learning_rate
        loss = step(inputs, targets)
        mx.eval(state)
        losses.append(loss.item())
        if (epoch + 1) % args.steps_per_report == 0:
            train_loss = np.mean(losses)
            toc = time.perf_counter()
            print(
                f"Iter {epoch + 1}: Train loss {train_loss:.3f}, "
                f"It/sec {args.steps_per_report / (toc - tic):.3f}"
            )
            losses = []
            tic = time.perf_counter()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BERT model")
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--steps_per_report", type=int, default=10)
    args = parser.parse_args()
    mx.set_default_device(mx.gpu)
    main(args)