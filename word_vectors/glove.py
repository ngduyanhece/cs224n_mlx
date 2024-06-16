"""
implementation of GloVe word vectors
original paper: https://nlp.stanford.edu/pubs/glove.pdf
"""
import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from functools import partial
import argparse
from data import DATA_SET, batch_iterate, get_vocab, visualize_embedding
# get the data and construct the co-occurrence matrix
def prepare_glove_data(data: str, window_size: int = 2):
    """
    prepare the data so that it could be used to train the GloVe algorithm
    args:
        - data: the text data in string format
        - window_size: the size of the context window
    returns:
        - a list of target-context pairs with the co-occurrence count
    """
    # tokenize the data
    tokens = data.split()
    # create a list of target-context pairs
    pairs = []
    for i in range(window_size, len(tokens) - window_size):
        target = tokens[i]
        context = [tokens[j] for j in range(i - window_size, i + window_size + 1) if i != j]
        for word in context:
            pairs.append((target, word))
    return pairs

def calculate_scores(pairs: list, word_to_index: dict):
    """
    calculate the co-occurrence scores for the target-context pairs
    args:
        - pairs: a list of target-context pairs
        - word_to_index: a dictionary mapping words to indices
    returns:
        - a list of target-context pairs with the co-occurrence scores
    """
    scores = {}
    for target, context in pairs:
        target_index = word_to_index[target]
        context_index = word_to_index[context]
        key = (target_index, context_index)
        scores[key] = scores.get(key, 0) + 1
    # normalize the scores
    max_score = max(scores.values())
    scores = {key: score / max_score for key, score in scores.items()}
    return scores

def convert_to_indices(pairs: list, word_to_index: dict):
    """
    convert the target-context pairs into indices
    args:
        - pairs: a list of target-context pairs
        - word_to_index: a dictionary mapping words to indices
    returns:
        - a list of target-context pairs with indices
    """
    data = []
    for target, context in pairs:
        target_index = word_to_index[target]
        context_index = word_to_index[context]
        data.append((target_index, context_index))
    return data

def prepare_data(data: str, window_size: int = 2):
    """
    prepare the data for training the GloVe model
    args:
        - data: the text data in string format
        - window_size: the size of the context window
    returns:
        - the data: the list of target-context pairs with indices and the co-occurrence scores
        - the word_to_index: a dictionary mapping words to indices
        - the index_to_word: a dictionary mapping indices to words
    """
    word_to_index, index_to_word = get_vocab(data)
    pairs = prepare_glove_data(data, window_size)
    scores = calculate_scores(pairs, word_to_index)
    pairs = convert_to_indices(pairs, word_to_index)
    data = [(pair, scores[pair]) for pair in pairs]
    return data, word_to_index, index_to_word

class GLove(nn.Module):
    """
    implementation of GloVe word vectors
    """
    def __init__(self, vocab_size, embedding_dim):
        super(GLove, self).__init__()
        self.target_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.bias_target = nn.Embedding(vocab_size, 1)
        self.bias_context = nn.Embedding(vocab_size, 1)

    def __call__(self, target_indices, context_indices):
        target = self.target_embeddings(target_indices)
        context = self.context_embeddings(context_indices)
        bias_target = mx.array.squeeze(self.bias_target(target_indices))
        bias_context = mx.array.squeeze(self.bias_context(context_indices))
        return mx.array.sum(target * context, axis=1) + bias_target + bias_context

def loss_fn(model, context, target, y):
    return nn.losses.cross_entropy(model(context, target), y, reduction="mean")

def main(args):
    # Example parameters
    batch_size = 16
    learning_rate = 0.001
    embedding_dim = 50

    data, word_to_index, index_to_word = prepare_data(DATA_SET)
    # map the data to mxnet arrays
    data, label = map(mx.array, zip(*data))
    # initialize the model
    model = GLove(len(word_to_index), embedding_dim)
    mx.eval(model.parameters())

    optimizer = optim.SGD(learning_rate)
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
       
        visualize_embedding(model.target_embeddings.weight, word_to_index)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GloVe word vectors")
    parser.add_argument("--visualize", action="store_true", help="Visualize the embeddings after training.")
    parser.add_argument("--num_epochs", type=int, default=1000, help="Number of epochs to train.")
    args = parser.parse_args()
    mx.set_default_device(mx.gpu)
    main(args)