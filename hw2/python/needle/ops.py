"""Operatpr table."""
# Global operator table.
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
        # WARNING
        # Dividing a float by an interger may introduce dtype mismatch. Fellows
        # on the forums reports `float32 / int` yields `float64`, although I
        # did not encounter this issue.
        #
        # Type alignment is pivotal in that optimizers shall not assign weights
        # of different type than the original one.
        return array_api.divide(a, self.scalar,
                                dtype=a.dtype)
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
        return (reshape(out_grad,
                        node.inputs[0].shape),) # a deliberate tuple
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
        #######################################################################
        # The original solution is not numerically stable.
        #
        # grad = divide(relu(node.inputs[0]), node.inputs[0])
        # return (multiply(out_grad, grad),)
        #######################################################################
        # There seems to be no numerically stable solution
        # that solely calls needle operations. assistance of
        # `array_api` is a must.
        node_input = node.inputs[0]
        return (multiply(out_grad,
                         Tensor(node_input.realize_cached_data() > 0,
                                device=node.device,
                                dtype=node.dtype,
                                required_grad=node.requires_grad)),) # a deliberate tuple
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)



class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        self.max = Z.max(axis=self.axes)
        if self.axes:
            # broadcast `z_max`` to shape of `Z`
            reduced_shape = list(Z.shape)
            for i in self.axes:
                reduced_shape[i] = 1
            self.reduced_shape = tuple(reduced_shape)

            return self.max + \
                   array_api.log(array_api.sum(array_api.exp(Z -
                                                             array_api.broadcast_to(array_api.reshape(self.max,
                                                                                                      self.reduced_shape),
                                                                                    Z.shape)),
                                               self.axes))
        else:
            self.reduced_shape = (1,) * len(Z.shape)
            return self.max + array_api.log(array_api.sum(array_api.exp(Z - self.max)))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        exp_z = exp(node.inputs[0] - reshape(Tensor(self.max,
                                                    dtype=node.inputs[0].dtype,
                                                    requires_grad=False),
                                             self.reduced_shape))
        # Needle summation does not preserve original shape.
        sum_z = reshape(summation(exp_z, self.axes),
                        self.reduced_shape)
        out_grad = reshape(out_grad, self.reduced_shape)
            
        return (multiply(out_grad,
                         divide(exp_z, sum_z)),) # a deliberate tuple
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
