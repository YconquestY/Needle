"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
from . import init
import numpy

from .backend_selection import array_api, NDArray

#from functools import reduce # for checking `TensorTuple`


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple([out_grad[i] for i in range(len(out_grad))])


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
                in_grad.append(init.zeros_like(value))
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
        return a + numpy.float32(self.scalar)

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


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
        return a * numpy.float32(self.scalar)

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
        #return array_api.power(a, self.scalar)
        return a ** self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return multiply(out_grad,
                        mul_scalar(array_api.power(node.inputs[0],
                                                   self.scalar - 1),
                                   self.scalar))
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        #return array_api.divide(a, b)
        return a / b
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
        #return array_api.divide(a, self.scalar,
        #                        dtype=a.dtype)
        return a / self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # quotient = dividend / divisor
        return divide_scalar(out_grad, self.scalar)
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
        else:
            order = order[ :-2] + [order[-1], order[-2]]
        #return array_api.transpose(a, axes=tuple(order))
        return a.permute(tuple(order))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return transpose(out_grad, axes=self.axes)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


### BEGIN YOUR SOLUTION
class Permute(TensorOp):
    def __init__(self, axes: tuple):
        self.axes = axes
    
    def compute(self, a):
        return a.permute(self.axes)
    
    def gradient(self, out_grad, node):
        raise NotImplementedError


def permute(a, axes):
    return Permute(axes)(a)
### END YOUR SOLUTION


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a, self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad,
                       node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)#.compact() # Why calling `compact`?

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        in_shape = node.inputs[0].shape
        # Tensors are not subscriptable in needle.
        # Call `reshape` alternatively to add axes.
        singleton = list(range(len(self.shape) - len(in_shape))) + \
                    [i for i in range(-1, -len(in_shape)-1, -1) if in_shape[i] == 1]
        return reshape(summation(out_grad,
                                 axes=tuple(singleton)),
                       in_shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if axes is None:
            self.axes = None
        elif isinstance(axes, int):
            self.axes = (axes,)
        else:
            self.axes = axes
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # Currently, Needle does not support multiple-axis summation in 1 pass.
        # Iterative summation along a single axis is adopted to bypass the
        # restriction. Multi-axis reduction will come in due time.
        #return array_api.summation(a, self.axes)
        if self.axes and len(self.axes) > 1:
            # Upon each summation, 1 dimension is reduced. Hence, the operation
            # must be performed in a descending order of axes, which implies
            # the axes must be positive and sorted.
            self.axes = sorted([axis if axis > 0 else len(self.axes) + axis for axis in self.axes])
            for axis in self.axes[ : :-1]:
                a = array_api.summation(a, axis)
            return a
        else:
            return array_api.summation(a, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        axes_shape = list(node.inputs[0].shape)
        # `axes` must be a tuple, even if it is a single integer.
        # Otherwise, `self.axes` is considered FALSE when assigned 0.
        if self.axes:
        #if self.axes is not None:
            for i in self.axes:
                axes_shape[i] = 1
        else:
            axes_shape = [1,] * len(axes_shape)
        return broadcast_to(reshape(out_grad, tuple(axes_shape)),
                            node.inputs[0].shape)
        ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        #return array_api.matmul(a, b)
        return a @ b
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
        return -a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return negate(out_grad)
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
        return divide(out_grad, node.inputs[0])
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return multiply(out_grad,
                        exp(node.inputs[0]))
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
        return multiply(out_grad,
                        Tensor(node_input.realize_cached_data() > 0,
                               device=node.device,
                               dtype=node.dtype,
                               required_grad=node.requires_grad))
        ### END YOUR SOLUTION


def relu(a):
    return ReLU()(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        if axes is None:
            self.axes = None
        elif isinstance(axes, int):
            self.axes = (axes,)
        else:
            self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION
        # 
        self.max = Z.max(axis=self.axes)
        # `self.axes` must be a tuple, even if it is a single integer.
        # Otherwise, `self.axes` is considered FALSE when assigned 0.
        #if self.axes:
        if self.axes is not None:
            # broadcast `z_max` to shape of `Z`
            reduced_shape = list(Z.shape)
            for i in self.axes:
                reduced_shape[i] = 1
            self.reduced_shape = tuple(reduced_shape)
        else:
            self.reduced_shape = (1,) * len(Z.shape)
        return self.max + \
               array_api.log(array_api.summation(array_api.exp(Z -
                                                               array_api.broadcast_to(array_api.reshape(self.max,
                                                                                                        self.reduced_shape),
                                                                                      Z.shape)),
                                                 self.axes))
        #else:
        #   self.reduced_shape = (1,) * len(Z.shape)
        #   return self.max + array_api.log(array_api.summation(array_api.exp(Z - self.max)))
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
            
        return multiply(out_grad,
                        divide(exp_z, sum_z))
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)


class Tanh(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.tanh(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * (init.ones(*out_grad.shape,
                                     device=out_grad.device,
                                     requires_grad=False) - power_scalar(tanh(node.inputs[0]), 2.))
        ### END YOUR SOLUTION


def tanh(a):
    return Tanh()(a)


class Stack(TensorOp):
    def __init__(self, axis: int):
        """
        Concatenates a sequence of arrays along a new dimension.
        Parameters:
        axis - dimension to concatenate along
        All arrays need to be of the same size.
        """
        self.axis = axis

    def compute(self, args):
        ### BEGIN YOUR SOLUTION
        #compare = lambda x, y : x == y
        # check array size
        #assert reduce(compare, map(lambda x : x.shape, args)), 'tensor dimension mismatch'
        shape = args[0].shape
        for tensor in args:
            if tensor.shape != shape:
                assert False, 'tensor dimension mismatch'
        # check array device
        #assert reduce(compare, map(lambda x : x.device, args)), 'tensor device mismatch'
        device = args[0].device
        for tensor in args:
            if tensor.device != device:
                assert False, 'tensor device mismatch'
        # stack tensors
        # see https://forum.dlsyscourse.org/t/stack-op-could-someone-share-some-idea-about-how-to-implement-stack-op/2804
        ndim = len(shape)
        out = array_api.empty(shape=shape[ :self.axis] + (len(args),) + shape[self.axis: ],
                              device=device)
        for i, tensor in enumerate(args):
            # see https://forum.dlsyscourse.org/t/q1-stack-issue-with-setitem/2887/2
            idx = (slice(None),) * self.axis + (i,) + (slice(None),) * (ndim - self.axis)
            out[idx] = tensor
        return out
        ### END YOUR SOLUTION

    def gradient(self, out_grad: Tensor, node):
        ### BEGIN YOUR SOLUTION
        return split(out_grad,
                     axis=self.axis)
        ### END YOUR SOLUTION


def stack(args, axis):
    return Stack(axis)(make_tuple(*args))


class Split(TensorTupleOp):
    def __init__(self, axis: int):
        """
        Splits a tensor along an axis into a tuple of tensors.
        (The "inverse" of Stack)
        Parameters:
        axis - dimension to split
        """
        self.axis = axis

    def compute(self, A):
        ### BEGIN YOUR SOLUTION
        shape = A.shape[ :self.axis] + A.shape[self.axis+1: ]
        out = []
        for i in range(A.shape[self.axis]):
            idx = (slice(None),) * self.axis + (i,) + (slice(None),) * (len(shape) - self.axis)
            # Do not forget to compact a tensor before reshaping it.
            out.append(array_api.reshape(A[idx].compact(),
                                         new_shape=shape))
        return tuple(out)
        ### END YOUR SOLUTION

    def gradient(self, out_grad: TensorTuple, node):
        ### BEGIN YOUR SOLUTION
        return stack(out_grad,
                     axis=self.axis)
        ### END YOUR SOLUTION


def split(a, axis):
    return Split(axis)(a)


class Flip(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a.flip(self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return flip(out_grad,
                    self.axes)
        ### END YOUR SOLUTION


def flip(a, axes):
    return Flip(axes)(a)



class Dilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        assert dilation >= 0, 'dilation must be non-negative'
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # positive dilation
        if self.dilation:
            stride = self.dilation + 1
            dilated_shape = tuple(size * stride
                                  if   axis in self.axes
                                  else size
                                  for  axis, size in enumerate(a.shape))
            out = array_api.full(shape=dilated_shape,
                                 fill_value=0.,
                                 device=a.device)
            idx = tuple(slice(None, None, stride)
                        if   axis in self.axes
                        else slice(None)
                        for  axis in range(a.ndim))
            out[idx] = a
            return out
        # 0 dilation
        return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return undilate(out_grad,
                        self.axes,
                        self.dilation)
        ### END YOUR SOLUTION


def dilate(a, axes, dilation):
    return Dilate(axes, dilation)(a)

class UnDilate(TensorOp):
    def __init__(self, axes: tuple, dilation: int):
        self.axes = axes
        assert dilation >= 0, 'dilation must be non-negative'
        self.dilation = dilation

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        # positive dilation
        if self.dilation:
            stride = self.dilation + 1
            undilated_shape = tuple(int(size / stride) if axis in self.axes
                                                          else size
                                                          for  axis, size in enumerate(a.shape))
            out = array_api.empty(shape=undilated_shape,
                                  device=a.device)
            idx = tuple(slice(None, None, (stride)) if   axis in self.axes
                                                    else slice(None)
                                                    for  axis in range(a.ndim))
            out = a[idx]
            return out
        # 0 dilation
        return a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return dilate(out_grad,
                      self.axes,
                      self.dilation)
        ### END YOUR SOLUTION


def undilate(a, axes, dilation):
    return UnDilate(axes, dilation)(a)


class Conv(TensorOp):
    def __init__(self, stride  : Optional[int] = 1,
                       padding : Optional[int] = 0,
                       dilation: Optional[int] = 1):
        self.stride   = stride
        self.padding  = padding
        self.dilation = dilation

    def compute(self, A, B):
        ### BEGIN YOUR SOLUTION
        '''feature map A convolved with kernel B
        A: NHWC
        B: HWIO        
        '''
        # Do not pad batch and channel dimensions.
        if self.padding:
            A = A.pad(((0, 0),
                       (self.padding, self.padding),
                       (self.padding, self.padding),
                       (0, 0)))
        N, H, W, C_in  = A.shape
        K, _, _, C_out = B.shape # square kernel by convention
        Ns, Hs, Ws, Cs = A.strides
        # im2col
        # Shape inference is tricky.
        # see https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        _H, _W = (H - self.dilation * (K-1) - 1) // self.stride + 1, \
                 (W - self.dilation * (K-1) - 1) // self.stride + 1
        inner_dim = K * K * C_in
        _A = A.as_strided(shape=(N, _H, _W, K, K, C_in),
                          strides=(Ns, Hs * self.stride,
                                       Ws * self.stride,
                                       Hs * self.dilation,
                                       Ws * self.dilation, Cs)).compact()
        # GEMM
        # It is necessary to compact `B`. This is bacause the backward pass of
        # `Conv` also involves convolution, which may flip the kernel.
        out = _A.reshape((-1, inner_dim)) @ B.compact().reshape((-1, C_out))
        return out.reshape((N, _H, _W, C_out))
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        A, B = node.inputs[0], node.inputs[1]
        N, H, W, C_in  = A.shape
        K, _, _, C_out = B.shape # square kernel by convention
                # w.r.t. feature map
        return (conv(dilate(out_grad,
                            axes=(1, 2),
                            dilation=self.stride),
                     transpose(flip(B, axes=(0, 1))),
                     padding=K-1-self.padding)[1: , 1: ], # full correlation
                # w.r.t. kernel
                permute(conv(permute(A, axes=(3, 1, 2, 0)),
                             permute(out_grad,
                                     axes=(1, 2, 0, 3)),
                             padding=self.padding),
                        axes=(1, 2, 0, 3)))
        ### END YOUR SOLUTION


def conv(a, b, stride=1, padding=0, dilation=1):
    return Conv(stride, padding, dilation)(a, b)
