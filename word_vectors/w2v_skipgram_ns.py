"""
implementation of word2vec skipgram with negative sampling
original paper: https://proceedings.neurips.cc/paper_files/paper/2013/file/9aa42b31882ec039965f3c4923ce901b-Paper.pdf
"""
# construct the positive and negative samples

import random

import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from functools import partial
import argparse

from data import DATA_SET, batch_iterate, get_vocab, visualize_embedding


def prepare_skipgram_ns_data(data: str, window_size: int = 2, num_neg_samples: int = 5):
    """
    prepare the data so that it could be train by the skipgram with negative sampling algorithm
    args:
        - data: the text data in string format
        - window_size: the size of the context window
        - num_neg_samples: the number of negative samples to be used
    returns:
        - a list of target-context and score pairs with 0 for negative samples and 1 for positive samples
    """
    # tokenize the data
    tokens = data.split()
    # create a list of target-context pairs
    pairs = []
    for i in range(window_size, len(tokens) - window_size):
        target = tokens[i]
        context = [tokens[j] for j in range(i - window_size, i + window_size + 1) if i != j]
        pairs.extend([([target, word], 1) for word in context])
        # create negative samples
        for _ in range(num_neg_samples):
            pairs.append(([target, random.choice(tokens)], 0))
    return pairs

def convert_to_indices(pairs: list, word_to_index: dict):
    """
    convert the context-target pairs into indices
    args:
        - pairs: a list of context-target pairs
        - word_to_index: a dictionary mapping words to indices
    returns:
        - a list of context-target pairs with indices
    """
    data = []
    for context, score in pairs:
        context_index = [word_to_index[word] for word in context]
        data.append((context_index, score))
    return data

class SkipGramNS(nn.Module):
    """
    implementation of word2vec using Skip-gram with negative sampling model
    """
    def __init__(self, vocab_size, embedding_dim):
        super(SkipGramNS, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)
    
    def __call__(self, context, target):
        target = self.embeddings(target)
        context = self.embeddings(context)
        score = target @ context.T
        return score

def loss_fn(model, context, target, y):
    return nn.losses.cross_entropy(model(context, target), y, reduction="mean")

def main(args):
    # Example parameters
    batch_size = 5
    learning_rate = 0.01
    embedding_dim = 50

    word_to_index, _ = get_vocab(DATA_SET)
    vocab_size = len(word_to_index)
    pairs = prepare_skipgram_ns_data(DATA_SET)
    data_set = convert_to_indices(pairs, word_to_index)
    # map the data to mxnet arrays
    data, label = map(mx.array, zip(*data_set))
    # initialize the model
    model = SkipGramNS(vocab_size, embedding_dim)
    mx.eval(model.parameters())

    optimizer = optim.SGD(learning_rate=learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    @partial(mx.compile, inputs=model.state, outputs=model.state)
    def step(context, target, y):
        loss, grads = loss_and_grad_fn(model, context, target, y)
        optimizer.update(model, grads)
        return loss

    for e in range(args.num_epochs):
        tic = time.perf_counter()
        for X, y in batch_iterate(batch_size, data, label):
            context, target = X[:, 0], X[:, 1]
            loss = loss_fn(model, context, target, y)
        toc = time.perf_counter()
        # print the loss and time taken for every 100 epochs
        if e % 100 == 0:
            print(
                f"Epoch {e}: Loss {loss}"
                f" Time {toc - tic:.3f} (s)"
            )
    if args.visualize:
       
        visualize_embedding(model.embeddings.weight, word_to_index)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Train a simple SkipGram with Negative Sampling Word Embedding with MLX.")
    parser.add_argument("--visualize", action="store_true", help="Visualize the embeddings after training.")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs to train.")
    args = parser.parse_args()
    mx.set_default_device(mx.gpu)
    main(args)