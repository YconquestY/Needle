"""The module.
"""
from typing import List
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(fan_in=self.in_features,
                                                     fan_out=self.out_features,
                                                     device=device, dtype=dtype,
                                                     requires_grad=True)) # Is this line necessary?
        if bias: # Do not learn a bias if `bias` is FALSE.
            self.bias = Parameter(ops.reshape(init.kaiming_uniform(fan_in=self.out_features,
                                                                   fan_out=1,
                                                                   device=device, dtype=dtype,
                                                                   requires_grad=True), # Is this line necessary?
                                              shape=(1, self.out_features)))
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Caveats
        # 1. There is no need to broadcast weights. This is because `MatMul`
        #    already considers the case of dimension mismatch. In practice,
        #    broadcasting weights leads to numerical errors during optimization.
        # 
        #    However, it is necessary to broadcast bias in that `EWiseAdd`
        #    applies to tensors of the same shape.
        # 2. Do not assign `self.bias` with the broadcast version, use a local
        #    variable instead.
        #
        #    Incorrect code:
        #        self.bias = ops.broadcast_to(self.bias,
        #                                     X.shape[ :-1] + (self.out_features,))
        #
        #    This irreversiblly modifies of the shape of `self.bias`.
        try:
            bias = self.bias
        except AttributeError:
            return X @ self.weight
        else:
            return X @ self.weight + ops.broadcast_to(bias,
                                                      X.shape[ :-1] + (self.out_features,))
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        return ops.reshape(X, (X.shape[0], -1))
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for fn in self.modules:
            x = fn(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        z_y = ops.summation(logits * init.one_hot(logits.shape[-1], y,
                                                  device=logits.device,
                                                  dtype=logits.dtype,
                                                  requires_grad=False),
                            axes=(-1,))
        return ops.summation(ops.logsumexp(logits,
                                           axes=(-1,)) - z_y) / logits.shape[0]
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, self.dim,
                                          device=device, dtype=dtype,
                                          requires_grad=True)) # Is this line necessary?
        self.bias = Parameter(init.zeros(1, self.dim,
                                         device=device, dtype=dtype,
                                         requires_grad=True)) # Is this line necessary?
        # The running estimates is updated during training. They do not
        # constitute the computational graph, hence are fixed in evaluation.
        self.running_mean = init.zeros(self.dim,
                                       device=device, dtype=dtype,
                                       requires_grad=False)
        self.running_var = init.ones(self.dim,
                                     device=device, dtype=dtype,
                                     requires_grad=False)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Needle does not support implicit broadcasting.
        # Do not forget to broadcast weight and bias.
        mean = ops.summation(x, axes=(-2,)) / x.shape[-2]
        shift = x - ops.broadcast_to(ops.reshape(mean,
                                                 (1, self.dim)),
                                     x.shape)
        var = ops.summation(shift ** 2. ,
                            axes=(-2,)) / x.shape[-2]
        if self.training:
            # `running_mean` and `running_var` do not constitute the
            # computational graph. Only update their `cached_data`.
            self.running_mean.data = (1. - self.momentum) * self.running_mean.data + self.momentum * mean.data
            self.running_var.data  = (1. - self.momentum) * self.running_var.data  + self.momentum * var.data
            std = ops.broadcast_to(ops.reshape((var + self.eps) ** .5,
                                               (1, self.dim)),
                                   x.shape)
            return ops.broadcast_to(self.weight,
                                    x.shape) * shift / std + ops.broadcast_to(self.bias,
                                                                              x.shape)
        else:
            x_hat = (x - ops.broadcast_to(ops.reshape(self.running_mean.data,
                                                      (1, self.dim)),
                                          x.shape)) / ops.broadcast_to(ops.reshape((self.running_var.data + self.eps) ** .5,
                                                                                   (1, self.dim)),
                                                                       x.shape)
            return ops.broadcast_to(self.weight.data,
                                    x.shape) * x_hat + ops.broadcast_to(self.bias.data,
                                                                        x.shape)
        ### END YOUR SOLUTION


class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, self.dim,
                                          device=device, dtype=dtype,
                                          requires_grad=True)) # Is this line necessary?
        self.bias = Parameter(init.zeros(1, self.dim,
                                         device=device, dtype=dtype,
                                         requires_grad=True)) # Is this line necessary?
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Needle does not support implicit broadcasting.
        # Do not forget to broadcast weight and bias.
        mean = ops.broadcast_to(ops.reshape(ops.summation(x, axes=(-1,)) / x.shape[-1],
                                            x.shape[ :-1] + (1,)),
                                x.shape)
        shift = x - mean
        std = ops.broadcast_to(ops.reshape((ops.summation(shift ** 2. ,
                                                          axes=(-1,)) / x.shape[-1] + self.eps) ** .5,
                                           x.shape[ :-1] + (1,)),
                               x.shape)
        return ops.broadcast_to(self.weight,
                                x.shape) * shift / std + ops.broadcast_to(self.bias,
                                                                          x.shape)
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape,
                              p=(1. - self.p),
                              device=x.device)
            return mask * x / (1. - self.p)
        else:
            return x
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return x + self.fn(x)
        ### END YOUR SOLUTION

class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class RNNCell(Module):
    def __init__(self, input_size, hidden_size,
                       bias=True, nonlinearity='tanh',
                       device=None, dtype="float32"):
        """
        Applies an RNN cell with tanh or ReLU nonlinearity.

        Parameters:
        input_size : The number of expected features in the input X
        hidden_size: The number of features in the hidden state h
        bias        : If FALSE, then the layer does not use bias weights
        nonlinearity: The non-linearity to use. Can be either 'tanh' or 'relu'.

        Variables:
        W_ih: The learnable input-hidden  weights of shape (input_size , hidden_size).
        W_hh: The learnable hidden-hidden weights of shape (hidden_size, hidden_size).
        bias_ih: The learnable input-hidden bias of shape (hidden_size,).
        bias_hh: The learnable hidden-hidden bias of shape (hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        self.hidden_size = hidden_size
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.W_ih = Parameter(init.rand(input_size, hidden_size,
                                        low=-np.sqrt(1. / hidden_size),
                                        high=np.sqrt(1. / hidden_size),
                                        device=device, dtype=dtype,
                                        requires_grad=True))
        self.W_hh = Parameter(init.rand(hidden_size, hidden_size,
                                        low=-np.sqrt(1. / hidden_size),
                                        high=np.sqrt(1. / hidden_size),
                                        device=device, dtype=dtype,
                                        requires_grad=True))
        # To save memory, do not initialize biases if not required.
        if bias:
            self.bias_ih = Parameter(init.rand(*(1, hidden_size),
                                               device=device, dtype=dtype,
                                               requires_grad=True))
            self.bias_hh = Parameter(init.rand(*(1, hidden_size),
                                               device=device, dtype=dtype,
                                               requires_grad=True))
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs:
        X of shape (bs, input_size ): Tensor containing input features
        h of shape (bs, hidden_size): Tensor containing the initial hidden state
            for each element in the batch. Defaults to zero if not provided.

        Outputs:
        h' of shape (bs, hidden_size): Tensor contianing the next hidden state
            for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        activation = ops.tanh if self.nonlinearity == 'tanh' else ops.relu
        batch_size = X.shape[0]
        # Instead of initializing a zero tensor when initial states are absent,
        # we use branches to avoid unnecessary memory allocation.
        if h and self.bias:
            return activation(X @ self.W_ih +
                              h @ self.W_hh + ops.broadcast_to(self.bias_ih,
                                                               (batch_size, self.hidden_size))
                                            + ops.broadcast_to(self.bias_hh,
                                                               (batch_size, self.hidden_size)))
        elif self.bias: # no initial state
            return activation(X @ self.W_ih + ops.broadcast_to(self.bias_ih,
                                                               (batch_size, self.hidden_size))
                                            + ops.broadcast_to(self.bias_hh,
                                                               (batch_size, self.hidden_size)))
        elif h: # no bias
            return activation(X @ self.W_ih + h @ self.W_hh)
        else:
            return activation(X @ self.W_ih)
        ### END YOUR SOLUTION


class RNN(Module):
    def __init__(self, input_size, hidden_size,
                       num_layers=1, bias=True, nonlinearity='tanh',
                       device=None, dtype="float32"):
        """
        Applies a multi-layer RNN with tanh or ReLU non-linearity to an input sequence.

        Parameters:
        input_size   - The number of expected features in the input x
        hidden_size  - The number of features in the hidden state h; identical
                       for all layers.
        num_layers   - Number of recurrent layers.
        nonlinearity - The non-linearity to use. Can be either 'tanh' or 'relu'.
        bias         - If FALSE, then the layer does not use bias weights.

        Variables:
        rnn_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, hidden_size) for k=0. Otherwise the shape is
            (hidden_size, hidden_size).
        rnn_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, hidden_size).
        rnn_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (hidden_size,).
        rnn_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (hidden_size,).
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        # used later in `forward`
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.rnn_cells = []
        for k in range(num_layers):
            input_size = input_size if k == 0 else hidden_size
            self.rnn_cells.append(RNNCell(input_size=input_size,
                                          hidden_size=hidden_size,
                                          bias=bias,
                                          nonlinearity=nonlinearity,
                                          device=device,
                                          dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, X, h0=None):
        """
        Inputs:
        X   of shape (seq_len, bs, input_size) containing the features of the
            input sequence.
        h_0 of shape (num_layers, bs, hidden_size) containing the initial hidden
            state for each element in the batch. Defaults to zeros if not provided.

        Outputs
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the RNN, for each t.
        h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        #
        # neat h and h0, X and H_l ?
        #
        H = ops.split(X, axis=0) # input
        h_n = []                 # final hidden state
        # provided initial state
        if h0:
            h0 = ops.split(h0, axis=0)
        for k in range(self.num_layers):
            # initialize hidden state
            # If not provided, initial states are set to `None` rather than a
            # zero tensor. This avoids unnecessary memory allocation and is
            # handled by multiple branches in `RNNCell`.
            h = h0[k] if h0 else None
            H_l = [] # cell outputs of current layer
            for t in range(X.shape[0]):
                h = self.rnn_cells[k](ops.tuple_get_item(H, t), h) # cell output of current time step
                H_l.append(h)
            
            H = ops.make_tuple(*H_l) # update input for next layer
            h_n.append(h)            # log output
        
        return ops.stack(H,   axis=0), \
               ops.stack(h_n, axis=0)
        ### END YOUR SOLUTION


class LSTMCell(Module):
    def __init__(self, input_size, hidden_size, bias=True, device=None, dtype="float32"):
        """
        A long short-term memory (LSTM) cell.

        Parameters:
        input_size - The number of expected features in the input X
        hidden_size - The number of features in the hidden state h
        bias - If False, then the layer does not use bias weights

        Variables:
        W_ih - The learnable input-hidden weights, of shape (input_size, 4*hidden_size).
        W_hh - The learnable hidden-hidden weights, of shape (hidden_size, 4*hidden_size).
        bias_ih - The learnable input-hidden bias, of shape (4*hidden_size,).
        bias_hh - The learnable hidden-hidden bias, of shape (4*hidden_size,).

        Weights and biases are initialized from U(-sqrt(k), sqrt(k)) where k = 1/hidden_size
        """
        super().__init__()
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (batch, input_size): Tensor containing input features
        h, tuple of (h0, c0), with
            h0 of shape (bs, hidden_size): Tensor containing the initial hidden state
                for each element in the batch. Defaults to zero if not provided.
            c0 of shape (bs, hidden_size): Tensor containing the initial cell state
                for each element in the batch. Defaults to zero if not provided.

        Outputs: (h', c')
        h' of shape (bs, hidden_size): Tensor containing the next hidden state for each
            element in the batch.
        c' of shape (bs, hidden_size): Tensor containing the next cell state for each
            element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class LSTM(Module):
    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        """
        Applies a multi-layer long short-term memory (LSTM) RNN to an input sequence.

        Parameters:
        input_size - The number of expected features in the input x
        hidden_size - The number of features in the hidden state h
        num_layers - Number of recurrent layers.
        bias - If False, then the layer does not use bias weights.

        Variables:
        lstm_cells[k].W_ih: The learnable input-hidden weights of the k-th layer,
            of shape (input_size, 4*hidden_size) for k=0. Otherwise the shape is
            (hidden_size, 4*hidden_size).
        lstm_cells[k].W_hh: The learnable hidden-hidden weights of the k-th layer,
            of shape (hidden_size, 4*hidden_size).
        lstm_cells[k].bias_ih: The learnable input-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        lstm_cells[k].bias_hh: The learnable hidden-hidden bias of the k-th layer,
            of shape (4*hidden_size,).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, X, h=None):
        """
        Inputs: X, h
        X of shape (seq_len, bs, input_size) containing the features of the input sequence.
        h, tuple of (h0, c0) with
            h_0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden state for each element in the batch. Defaults to zeros if not provided.
            c0 of shape (num_layers, bs, hidden_size) containing the initial
                hidden cell state for each element in the batch. Defaults to zeros if not provided.

        Outputs: (output, (h_n, c_n))
        output of shape (seq_len, bs, hidden_size) containing the output features
            (h_t) from the last layer of the LSTM, for each t.
        tuple of (h_n, c_n) with
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden state for each element in the batch.
            h_n of shape (num_layers, bs, hidden_size) containing the final hidden cell state for each element in the batch.
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype="float32"):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Parameters:
        num_embeddings (int) - Size of the dictionary
        embedding_dim (int) - The size of each embedding vector

        Variables:
        weight - The learnable weights of shape (num_embeddings, embedding_dim)
            initialized from N(0, 1).
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        """
        Maps word indices to one-hot vectors, and projects to embedding vectors

        Input:
        x of shape (seq_len, bs)

        Output:
        output of shape (seq_len, bs, embedding_dim)
        """
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION
