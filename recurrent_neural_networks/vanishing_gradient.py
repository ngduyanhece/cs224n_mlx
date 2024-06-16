"""
we illustrate the vanishing gradient problem in RNNs by training a simple RNN to learn a simple sequence of numbers.
read more about the difficulties of training RNNs in the following paper: https://arxiv.org/pdf/1211.5063.pdf
"""
import argparse
from functools import partial
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from rnn import RNN

from word_vectors.data import batch_iterate


def loss_fn(model, X, hidden, y):
    output, hidden = model(X, hidden)
    return nn.losses.cross_entropy(output, y, reduction="mean"), hidden

def main(args):
    # Example parameters
    data = mx.random.normal(shape=(100, 5))
    labels = mx.random.normal(shape=(100, 1))

    input_size = data.shape[1]
    hidden_size = 10
    output_size = 1
    learning_rate = 0.001
    batch_size = 4

    model = RNN(input_size, hidden_size, output_size)

    mx.eval(model.parameters())

    optimizer = optim.SGD(learning_rate=learning_rate)
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    @partial(mx.compile, inputs=model.state, outputs=model.state)
    def step(X, hidden, y):
        loss_hidden, grads = loss_and_grad_fn(model, X, hidden, y)
        optimizer.update(model, grads)
        loss, hidden = loss_hidden
        return loss, grads, hidden

    hidden_state = model.init_hidden(batch_size)
    for e in range(args.num_epochs):
        tic = time.perf_counter()
        for X, y in batch_iterate(batch_size, data, labels):
            loss, grads, hidden = step(X, hidden_state, y)
            hidden_state = hidden
            sum_grads = mx.sum(grads['i2h']['weight'])
        toc = time.perf_counter()
        # print the loss and time taken for every 100 epochs
        if e % 100 == 0:
            print(
                f"Epoch {e}: Loss {loss} - Gradient {sum_grads}"
                f" Time {toc - tic:.3f} (s)"
            )
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Vanishing gradient problem in RNNs")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs to train.")
    args = parser.parse_args()
    main(args)