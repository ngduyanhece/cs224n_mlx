"""
implementation of a LSTM
read more about LSTMs here: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
"""
import mlx.core as mx
import mlx.nn as nn

class LSTM(nn.Module):
    """
    implementation of a LSTM
    """
    def __init__(self, input_size, hidden_size):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        # Input gate
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        # Forget gate
        self.f2h = nn.Linear(input_size + hidden_size, hidden_size)
        # Output gate
        self.o2h = nn.Linear(input_size + hidden_size, hidden_size)
        # Cell state
        self.c2h = nn.Linear(input_size + hidden_size, hidden_size)
        # Output
        self.h2o = nn.Linear(hidden_size, input_size)

    def __call__(self, input, hidden):
        h_prev, c_prev = hidden
        # concatenate the input and hidden state
        combined = mx.concatenate([input, h_prev], axis=1)
        # calculate the input gate
        i = nn.sigmoid(self.i2h(combined))
        # calculate the forget gate
        f = nn.sigmoid(self.f2h(combined))
        # calculate the output gate
        o = nn.sigmoid(self.o2h(combined))
        # calculate the cell state
        c = nn.tanh(self.c2h(combined))
        c = f * c_prev + i * c
        # calculate the hidden state
        h = o * nn.tanh(c)
        # calculate the output
        output = self.h2o(h)
        return output, (h, c)

    def init_hidden(self, batch_size):
        return (mx.zeros((batch_size, self.hidden_size)), mx.zeros((batch_size, self.hidden_size)))
    
def main():
    input_size = 10
    hidden_size = 20
    lstm = LSTM(input_size, hidden_size)

    # Example input (batch_size, input_size)
    input = mx.random.normal(shape=(5, input_size))
    hidden = lstm.init_hidden(5)

    # forward pass
    output, hidden = lstm(input, hidden)
    print(hidden)

if __name__ == "__main__":
    main()