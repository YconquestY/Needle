"""Operator and gradient implementations."""
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp

from .autograd import TensorTuple, TensorTupleOp

import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad,)


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (multiply(out_grad,
                         mul_scalar(array_api.power(node.inputs[0],
                                                    self.scalar - 1),
                                    self.scalar)),) # a deliberate tuple
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # quotient = dividend / divisor
        dividend, divisor = node.inputs
        return divide(out_grad, divisor), \
               negate(multiply(out_grad,
                               divide(dividend,
                                      power_scalar(divisor, 2))))
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a, self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # quotient = dividend / divisor
        return (divide_scalar(out_grad, self.scalar),) # a deliberate tuple
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # `Transpose` behaves differently from `numpy.transpose`
        # in terms of input and default axes permutated.
        order = list(range(len(a.shape)))
        if self.axes:
            order[self.axes[0]], order[self.axes[1]] = order[self.axes[1]], order[self.axes[0]]
            return array_api.transpose(a, axes=order)
        else:
            order = order[ :-2] + [order[-1], order[-2]]
            return array_api.transpose(a, axes=order)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (transpose(out_grad, axes=self.axes),) # a deliberate tuple
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (reshape(out_grad, node.inputs[0].shape),) # a deliberate tuple
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape = node.inputs[0].shape
        # Tensors are not subscriptable in needle.
        # Call `reshape` alternatively to add axes.
        singleton = list(range(len(self.shape) - len(in_shape))) + \
                    [i for i in range(-1, -len(in_shape)-1, -1) if in_shape[i] == 1]
        return (reshape(summation(out_grad,
                                 axes=tuple(singleton)),
                        in_shape),) # a deliberate tuple
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Tensors are not subscriptable in needle.
        # Call `reshape` alternatively to add axes.
        axes_shape = list(node.inputs[0].shape)
        if self.axes:
            for i in self.axes:
                axes_shape[i] = 1
        else:
            axes_shape = [1] * len(axes_shape)
        return (broadcast_to(reshape(out_grad, axes_shape),
                             node.inputs[0].shape),) # a deliberate tuple
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.matmul(a, b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs, rhs = node.inputs
        out_shape, lhs_shape, rhs_shape = out_grad.shape, lhs.shape, rhs.shape
        
        return      matmul(out_grad, transpose(rhs)) if len(lhs_shape) == len(out_shape) \
               else summation(matmul(out_grad, transpose(rhs)),                          \
                              axes=tuple(range(len(out_shape) - len(lhs_shape)))),       \
                                                                                         \
                    matmul(transpose(lhs), out_grad) if len(rhs_shape) == len(out_shape) \
               else summation(matmul(transpose(lhs), out_grad),                          \
                              axes=tuple(range(len(out_shape) - len(rhs_shape))))
        ### END YOUR SOLUTION


def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.negative(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (negate(out_grad),) # a deliberate tuple
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (divide(out_grad, node.inputs[0]),) # a deliberate tuple
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return (multiply(out_grad,
                         exp(node.inputs[0])),) # a deliberate tuple
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        grad = divide(relu(node.inputs[0]), node.inputs[0])
        return (multiply(out_grad, grad),) # a deliberate tuple
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSoftmax(TensorOp):
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        raise NotImplementedError()
        ### END YOUR SOLUTION


def logsoftmax(a):
    return LogSoftmax()(a)


# additional helper functions
def full(
    shape, fill_value, *, rand={}, dtype="float32", device=None, requires_grad=False
):
    # numpy do not need device argument
    kwargs = {"device": device} if array_api is not numpy else {}
    device = device if device else cpu()

    if not rand or "dist" not in rand:
        arr = array_api.full(shape, fill_value, dtype=dtype, **kwargs)
    else:
        if rand["dist"] == "normal":
            arr = array_api.randn(
                shape, dtype, mean=rand["mean"], std=rand["std"], **kwargs
            )
        if rand["dist"] == "binomial":
            arr = array_api.randb(
                shape, dtype, ntrials=rand["trials"], p=rand["prob"], **kwargs
            )
        if rand["dist"] == "uniform":
            arr = array_api.randu(
                shape, dtype, low=rand["low"], high=rand["high"], **kwargs
            )

    return Tensor.make_const(arr, requires_grad=requires_grad)


def zeros(shape, *, dtype="float32", device=None, requires_grad=False):
    return full(shape, 0, dtype=dtype, device=device, requires_grad=requires_grad)


def randn(
    shape, *, mean=0.0, std=1.0, dtype="float32", device=None, requires_grad=False
):
    return full(
        shape,
        0,
        rand={"dist": "normal", "mean": mean, "std": std},
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def randb(shape, *, n=1, p=0.5, dtype="float32", device=None, requires_grad=False):
    return full(
        shape,
        0,
        rand={"dist": "binomial", "trials": n, "prob": p},
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def randu(shape, *, low=0, high=1, dtype="float32", device=None, requires_grad=False):
    return full(
        shape,
        0,
        rand={"dist": "uniform", "low": low, "high": high},
        dtype=dtype,
        device=device,
        requires_grad=requires_grad,
    )


def zeros_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return full(
        array.shape, 0, dtype=array.dtype, device=device, requires_grad=requires_grad
    )


def ones_like(array, *, device=None, requires_grad=False):
    device = device if device else array.device
    return full(
        array.shape, 1, dtype=array.dtype, device=device, requires_grad=requires_grad
    )
