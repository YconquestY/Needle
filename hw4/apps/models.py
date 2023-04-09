import sys
sys.path.append('./python')
import needle as ndl
import needle.nn as nn
import math
import numpy as np
np.random.seed(0)


class ResNet9(ndl.nn.Module):
    def __init__(self, device=None, dtype="float32"):
        super().__init__()
        ### BEGIN YOUR SOLUTION ###
        raise NotImplementedError() ###
        ### END YOUR SOLUTION

    def forward(self, x):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class LanguageModel(nn.Module):
    def __init__(self, embedding_size, output_size, hidden_size,
                       num_layers=1, seq_model='rnn',
                       bias=True, #nonlinearity='tanh',
                       device=None, dtype="float32"):
        """
        Consists of an embedding layer, a sequence model (either RNN or LSTM), and a
        linear layer.
        Parameters:
        output_size   : Size of dictionary
        embedding_size: Size of embeddings
        hidden_size   : The number of features in the hidden state of LSTM or RNN
        seq_model : 'rnn' or 'lstm', whether to use RNN or LSTM
        num_layers: Number of layers in RNN or LSTM
        """
        super(LanguageModel, self).__init__()
        ### BEGIN YOUR SOLUTION
        # used later in `forward`
        self.model = seq_model
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(num_embeddings=output_size,
                                      embedding_dim=embedding_size,
                                      device=device,
                                      dtype=dtype)
        model = nn.RNN if seq_model == 'rnn' else nn.LSTM
        self.seq_model = model(input_size=embedding_size,
                               hidden_size=hidden_size,
                               num_layers=num_layers,
                               bias=bias,
                               #nonlinearity=nonlinearity, # `nonlinearity` not applicable to LSTM
                               device=device,
                               dtype=dtype)
        self.word = nn.Linear(in_features=hidden_size,
                              out_features=output_size,
                              bias=bias,
                              device=device,
                              dtype=dtype)
        ### END YOUR SOLUTION

    def forward(self, x, h=None):
        """
        Given sequence (and the previous hidden state if given), returns probabilities of next word
        (along with the last hidden state from the sequence model).
        Inputs:
        x of shape (seq_len, bs)
        h of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        Returns (out, h)
        out of shape (seq_len x bs  , output_size)
        h   of shape (num_layers, bs, hidden_size) if using RNN,
            else h is tuple of (h0, c0), each of shape (num_layers, bs, hidden_size)
        """
        ### BEGIN YOUR SOLUTION
        x = self.embedding(x)
        # `x`: (`seq_len`, `bs`, `embedding_size`)
        out, h = self.seq_model(x, h)
        # `out`: (`seq_len`, `bs`, `hidden_size`)
        # `h`:
        #     RNN : (`num_layers`, `bs`, `hidden_size`)
        #     LSTM: (`h0`: (`num_layers`, `bs`, `hidden_size`),
        #            `c0`: (`num_layers`, `bs`, `hidden_size`))
        # Needle does not support "stacked" matrix multiplication.
        out = out.reshape(shape=(-1, out.shape[-1]))
        # `out`: (`seq_len` x `bs`, `hidden_size`)
        out = self.word(out)
        # `out`: (`seq_len` x `bs`, `output_size`)
        return out, h
        ### END YOUR SOLUTION


if __name__ == "__main__":
    model = ResNet9()
    x = ndl.ops.randu((1, 32, 32, 3), requires_grad=True)
    model(x)
    cifar10_train_dataset = ndl.data.CIFAR10Dataset("data/cifar-10-batches-py", train=True)
    train_loader = ndl.data.DataLoader(cifar10_train_dataset, 128, ndl.cpu(), dtype="float32")
    print(dataset[1][0].shape)