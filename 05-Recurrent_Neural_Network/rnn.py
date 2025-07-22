import numpy as np

class SimpleRNN:
    def __init__(self, input_size, hidden_size, output_size, seq_len):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.seq_len = seq_len

        # Weight matrices and biases
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01  # input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # hidden to hidden (recurrent)
        self.Why = np.random.randn(output_size, hidden_size) * 0.01  # hidden to output

        self.bh = np.zeros((hidden_size, 1))  # hidden bias
        self.by = np.zeros((output_size, 1))  # output bias

    def forward(self, inputs):
        """
        inputs: list of input vectors (each of shape input_size x 1)
        Returns: outputs and hidden states for all time steps
        """
        h = np.zeros((self.hidden_size, 1))  # h0 initialized to zero
        hs, ys = {}, {}
        hs[-1] = h

        for t in range(self.seq_len):
            x = inputs[t]
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, hs[t - 1]) + self.bh)
            y = np.dot(self.Why, h) + self.by

            hs[t] = h
            ys[t] = y

        return ys, hs

    def backward(self, inputs, targets, ys, hs, learning_rate=0.001):
        """
        Perform Backpropagation Through Time (BPTT)
        """
        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dWhy = np.zeros_like(self.Why)
        dbh = np.zeros_like(self.bh)
        dby = np.zeros_like(self.by)

        dh_next = np.zeros_like(hs[0])
        loss = 0

        for t in reversed(range(self.seq_len)):
            dy = ys[t] - targets[t]  # dL/dy
            dWhy += np.dot(dy, hs[t].T)
            dby += dy
            loss += 0.5 * np.sum(dy ** 2)

            dh = np.dot(self.Why.T, dy) + dh_next
            dtanh = (1 - hs[t] ** 2) * dh

            dbh += dtanh
            dWxh += np.dot(dtanh, inputs[t].T)
            dWhh += np.dot(dtanh, hs[t - 1].T)
            dh_next = np.dot(self.Whh.T, dtanh)

        # Clip gradients to prevent exploding gradients
        for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
            np.clip(dparam, -1, 1, out=dparam)

        # Update weights
        self.Wxh -= learning_rate * dWxh
        self.Whh -= learning_rate * dWhh
        self.Why -= learning_rate * dWhy
        self.bh -= learning_rate * dbh
        self.by -= learning_rate * dby

        return loss

    def train(self, X, Y, epochs=1000, learning_rate=0.001):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                inputs = X[i]
                targets = Y[i]

                ys, hs = self.forward(inputs)
                loss = self.backward(inputs, targets, ys, hs, learning_rate)
                total_loss += loss
            if epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")

if __name__ == '__main__':
    # Dummy sequential input: let's say input_size=2, output_size=1
    input_size = 2
    hidden_size = 4
    output_size = 1
    seq_len = 3

    # Example training data
    X_train = [
        [np.random.rand(input_size, 1) for _ in range(seq_len)] for _ in range(10)
    ]
    Y_train = [
        [np.random.rand(output_size, 1) for _ in range(seq_len)] for _ in range(10)
    ]

    # Initialize and train the RNN
    rnn = SimpleRNN(input_size, hidden_size, output_size, seq_len)
    rnn.train(X_train, Y_train, epochs=500, learning_rate=0.01)
