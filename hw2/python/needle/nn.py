"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np

from functools import reduce # cumulative product
from operator import mul     # of an iterable


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
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        # weight initialization
        self.weight = Parameter(init.kaiming_uniform(fan_in=self.in_features,
                                                     fan_out=self.out_features,
                                                     device=device, dtype=dtype))
        # bias initialization
        self.use_bias = bias
        if self.use_bias:
            self.bias = Parameter(ops.reshape(init.kaiming_uniform(fan_in=self.out_features,
                                                                   fan_out=1,
                                                                   device=device, dtype=dtype),
                                              shape=(1, self.out_features)))
        #print(self.bias.shape)
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # needle tensors do not support the `ndim` attribute.
        # broadcast weights
        if len(X.shape) > 2:
            self.weight = ops.broadcast_to(ops.reshape(self.weight,
                                                       (1,) + self.weight.shape),
                                           (X.shape[0],) + self.weight.shape)
        # broadcasr bias: random initialization replicated?
        if self.use_bias:
            if len(X.shape) > 2:
                self.bias = ops.broadcast_to(ops.reshape(self.bias,
                                                         (1,) + self.bias.shape),
                                             tuple(X.shape[ :2]) + (self.out_features,))
            else:
                self.bias = ops.broadcast_to(self.bias,
                                             (X.shape[0], self.out_features))
        
        if self.use_bias:
            return X @ self.weight + self.bias
        else:
            return X @ self.weight
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


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        for module in self.modules:
            x = module(x)
        return x
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        z_y = ops.summation(logits * init.one_hot(logits.shape[-1], y,
                                                  device=logits.device,
                                                  dtype=logits.dtype),
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
                                          device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(1, self.dim,
                                         device=device, dtype=dtype))
        # The following instance variables shall exist even though the layer
        # is initialized for training.
        self.running_mean = Parameter(init.zeros(self.dim,
                                                 device=device, dtype=dtype))
        self.running_var = Parameter(init.ones(self.dim,
                                               device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Needle does not support implicit broadcasting.
        mean = ops.summation(x, axes=(-2,)) / x.shape[-2]
        self.running_mean = (1. - self.momentum) * self.running_mean + self.momentum * mean
        
        shift = x - ops.broadcast_to(ops.reshape(mean,
                                                 (1, self.dim)),
                                     x.shape)
        var = ops.summation(shift ** 2. ,
                            axes=(-2,)) / x.shape[-2]
        self.running_var = (1. - self.momentum) * self.running_var  + self.momentum * var

        if self.training:
            std = ops.broadcast_to(ops.reshape((ops.summation(shift ** 2. ,
                                                              axes=(-2,)) / x.shape[-2] + self.eps) ** .5,
                                               (1, self.dim)),
                                   x.shape)
            return self.weight * shift / std + self.bias
        else:
            return (x - ops.broadcast_to(ops.reshape(self.running_mean,
                                                     (1, self.dim)),
                                         x.shape)) / (ops.broadcast_to(ops.reshape(self.running_var,
                                                                                   (1, self.dim)),
                                                                       x.shape) + self.eps) ** .5
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.ones(1, self.dim,
                                          device=device, dtype=dtype))
        self.bias = Parameter(init.zeros(1, self.dim,
                                         device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # Needle does not support implicit broadcasting.
        mean = ops.broadcast_to(ops.reshape(ops.summation(x, axes=(-1,)) / x.shape[-1],
                                            (x.shape[-2], 1)),
                                x.shape)
        shift = x - mean
        std = ops.broadcast_to(ops.reshape((ops.summation(shift ** 2. ,
                                                          axes=(-1,)) / x.shape[-1] + self.eps) ** .5,
                                           (x.shape[-2], 1)),
                               x.shape)
        return self.weight * shift / std + self.bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mask = init.randb(*x.shape, p=self.p, device=x.device)
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
