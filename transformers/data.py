"""
this file load the data and prepare it for training the transformer language model
the majority of the script is taken from the official mlx tutorial https://github.com/ml-explore/mlx-examples/tree/main/transformer_lm
"""

import itertools

import numpy as np


def dummy_data():
    """
    dummy data
    """
    lines = [
        "The history of the world is the memory of the past",
        "The human history began with the invention of writing",
        "The writing of history is an important part of human culture",
    ]
    vocab = set(t for line in lines for t in line.split())
    eos = "<eos>"
    vocab.add(eos)
    vocab = {v: i for i, v in enumerate(vocab)}

    data_set = np.array(
        [vocab[w] for line in lines for w in itertools.chain(line.split(), [eos])],
        dtype=np.int32,
    )

    return vocab, data_set

def to_samples(context_size, dataset):
    """
    convert the dataset into target-context pairs
    """
    tokens = dataset.size
    window_size = context_size + 1  # include target
    samples = tokens - window_size + 1
    X = np.lib.stride_tricks.as_strided(
        dataset,
        shape=(samples, window_size),
        strides=(dataset.itemsize, dataset.itemsize),
    )
    return X[:, :-1], X[:, 1:]


def iterate_batches(batch_size, context_size, dataset):
    """
    iterate over the dataset in batches
    """
    inputs, targets = to_samples(context_size, dataset)
    s = 0
    while True:
        if s == 0:
            # Reset permutation:
            perm = np.random.permutation(inputs.shape[0])
        ids = perm[s: s + batch_size]
        yield inputs[ids], targets[ids]
        s += batch_size
        if s >= inputs.shape[0]:
            s = 0