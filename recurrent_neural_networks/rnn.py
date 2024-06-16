"""
implementation of a simple RNN
"""
import mlx.core as mx
import mlx.nn as nn

class RNN(nn.Module):
    """
    implementation of a simple RNN
    """
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        # input to hidden
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # hidden to output
        self.h2o = nn.Linear(hidden_size, output_size)
    
    def __call__(self, input, hidden):
        # concatenate the input and hidden state
        combined = mx.concatenate([input, hidden], axis=1)
        # calculate the hidden state
        hidden = self.i2h(combined)
        # calculate the output
        output = self.h2o(hidden)
        output = nn.tanh(output)
        return output, hidden

    def init_hidden(self, batch_size=1):
        return mx.zeros((batch_size, self.hidden_size))


def main():
    input_size = 3
    hidden_size = 10
    output_size = 3
    num_embeddings = 1
    dims = input_size

    rnn = RNN(input_size, hidden_size, output_size)

    # Example input
    input = mx.random.normal(shape=(num_embeddings, input_size))
    hidden = rnn.init_hidden()
    # forward pass
    output, hidden = rnn(input, hidden)
    print(output)

if __name__ == "__main__":
    main()