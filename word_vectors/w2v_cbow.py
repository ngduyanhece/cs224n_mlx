"""
implementation of word2vec using CBOW model
original paper: https://arxiv.org/pdf/1301.3781.pdf
"""
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from functools import partial
import argparse

from data import DATA_SET, batch_iterate, get_vocab, visualize_embedding


def prepare_cbow_data(data: str, window_size: int = 2):
    """
    prepare the data so that it could be train by the cbow algorithm
    args:
        - data: the text data in string format
        - window_size: the size of the context window
    returns:
        - a list of context-target pairs
    """
    # tokenize the data
    tokens = data.split()
    # create a list of context-target pairs
    pairs = []
    for i in range(window_size, len(tokens) - window_size):
        context = [tokens[j] for j in range(i - window_size, i + window_size + 1) if i != j]
        target = tokens[i]
        pairs.append((context, target))
    return pairs

# convert the context-target pairs into indices
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
    for context, target in pairs:
        context_indices = [word_to_index[word] for word in context]
        target_index = word_to_index[target]
        data.append((context_indices, target_index))
    return data


class CBOW(nn.Module):
    """
    implementation of word2vec using CBOW model
    """
    def __init__(self, vocab_size, embedding_dim):
        super(CBOW, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear = nn.Linear(embedding_dim, vocab_size)

    def __call__(self, context):
        context = self.embeddings(context)
        context = mx.array.mean(context, axis=1)
        out = self.linear(context)
        return out

def loss_fn(model, X, y):
    return nn.losses.cross_entropy(model(X), y, reduction="mean")

def main(args):
    # Example parameters
    batch_size = 5
    learning_rate = 0.01
    embedding_dim = 50

    word_to_index, _ = get_vocab(DATA_SET)
    vocab_size = len(word_to_index)
    pairs = prepare_cbow_data(DATA_SET)
    data_set = convert_to_indices(pairs, word_to_index)
    # map the data to mxnet arrays
    data, label = map(mx.array, zip(*data_set))
    # initialize the model
    model = CBOW(vocab_size, embedding_dim)
    mx.eval(model.parameters())

    optimizer = optim.SGD(learning_rate=learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    @partial(mx.compile, inputs=model.state, outputs=model.state)
    def step(X, y):
        loss, grads = loss_and_grad_fn(model, X, y)
        optimizer.update(model, grads)
        return loss

    for e in range(args.num_epochs):
        tic = time.perf_counter()
        for X, y in batch_iterate(batch_size, data, label):
            loss = step(X, y)
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
    parser = argparse.ArgumentParser("Train a simple CBOW Word Embedding with MLX.")
    parser.add_argument("--visualize", action="store_true", help="Visualize the embeddings after training.")
    parser.add_argument("--num_epochs", type=int, default=2000, help="Number of epochs to train.")
    args = parser.parse_args()
    mx.set_default_device(mx.gpu)
    main(args)
