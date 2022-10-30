"""Core data structures."""
import needle
from typing import List, Optional, NamedTuple, Tuple, Union, Dict
from collections import namedtuple
import numpy
from needle import init

from operator  import add    # for helper `sum_node_list` to sum a list of tensors
from functools import reduce # without indroducing `dtype mismatch`

# needle version
LAZY_MODE = False
TENSOR_COUNTER = 0

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api

NDArray = numpy.ndarray


class Device:
    """Indicates the device supporting an NDArray."""


class CPUDevice(Device):
    """Represents data that sits in CPU"""
    # return a string representing the object (as the developer wants to see it)
    def __repr__(self):
        return "needle.cpu()"
    # An unhashbale object cannot be put in a set.
    def __hash__(self):
        return self.__repr__().__hash__()
    # A hashable object must also implement `__eq__`.
    def __eq__(self, other):
        return isinstance(other, CPUDevice)

    def enabled(self):
        return True

    def zeros(self, *shape, dtype="float32"):
        return numpy.zeros(shape, dtype=dtype)

    def ones(self, *shape, dtype="float32"):
        return numpy.ones(shape, dtype=dtype)

    def randn(self, *shape):
        # note: numpy doesn't support types within standard random routines, and 
        # .astype("float32") does work if we're generating a singleton
        return numpy.random.randn(*shape) 

    def rand(self, *shape):
        # note: numpy doesn't support types within standard random routines, and 
        # .astype("float32") does work if we're generating a singleton
        return numpy.random.rand(*shape)

    def one_hot(self, n, i, dtype="float32"):
        return numpy.eye(n, dtype=dtype)[i]


def cpu():
    """Return cpu device"""
    return CPUDevice()


def all_devices():
    """return a list of all available devices"""
    return [cpu()]


class Op:
    """Operator definition."""

    def __call__(self, *args):
        raise NotImplementedError()

    def compute(self, *args: Tuple[NDArray]):
        """Calculate forward pass of operator.

        Parameters
        ----------
        input: np.ndarray
            A list of input arrays to the function

        Returns
        -------
        output: ndarray
            Array output of the operation

        """
        raise NotImplementedError()

    def gradient(self,
                 out_grad: "Value",
                 node    : "Value") -> Union["Value", Tuple["Value"]]:
        """Compute partial adjoint for each input value for a given output adjoint.

        Parameters
        ----------
        out_grad: Value
            The adjoint w.r.t. to the output value.

        node: Value
            The value node of forward evaluation.

        Returns
        -------
        input_grads: Value or Tuple[Value]
            A list containing partial gradient adjoints to be propagated to
            each of the input node.
        """
        raise NotImplementedError()

    def gradient_as_tuple(self,
                          out_grad: "Value",
                          node    : "Value") -> Tuple["Value"]:
        """ Convenience method to always return a tuple from gradient call"""
        output = self.gradient(out_grad, node)
        if isinstance(output, tuple):
            return output
        elif isinstance(output, list):
            return tuple(output)
        else:
            return (output,)


class TensorOp(Op):
    """ Op class specialized to output tensors, will be alterate subclasses for other structures """

    def __call__(self, *args):
        return Tensor.make_from_op(self, args)


class TensorTupleOp(Op):
    """Op class specialized to output TensorTuple"""

    def __call__(self, *args):
        return TensorTuple.make_from_op(self, args)


class Value:
    """A value in the computational graph."""

    # trace of computational graph
    op: Optional[Op]
    inputs: List["Value"]
    # The following fields are cached fields for
    # dynamic computation
    cached_data: NDArray
    requires_grad: bool

    def realize_cached_data(self):
        """Run compute to realize the cached data"""
        # avoid recomputation
        if self.cached_data is not None:
            return self.cached_data
        # note: data implicitly calls realized cached data
        self.cached_data = self.op.compute(
            *[x.realize_cached_data() for x in self.inputs]
        )
        #self.cached_data
        return self.cached_data

    def is_leaf(self):
        return self.op is None

    def __del__(self):
        global TENSOR_COUNTER
        TENSOR_COUNTER -= 1

    def _init(self,
              op: Optional[Op],
              inputs: List["Tensor"],
              *,
              num_outputs: int = 1,
              cached_data: List[object] = None,
              requires_grad: Optional[bool] = None):
        global TENSOR_COUNTER
        TENSOR_COUNTER += 1
        if requires_grad is None:
            requires_grad = any(x.requires_grad for x in inputs)
        self.op = op
        self.inputs = inputs
        self.num_outputs = num_outputs
        self.cached_data = cached_data
        self.requires_grad = requires_grad

    @classmethod
    def make_const(cls, data, *, requires_grad=False):
        value = cls.__new__(cls)
        value._init(
            None,
            [],
            cached_data=data,
            requires_grad=requires_grad,
        )
        return value

    @classmethod
    def make_from_op(cls, op: Op, inputs: List["Value"]):
        value = cls.__new__(cls)
        value._init(op, inputs)

        if not LAZY_MODE:
            if not value.requires_grad:
                return value.detach()
            value.realize_cached_data()
        return value


class Tensor(Value):
    grad: "Tensor"

    def __init__(
        self,
        array,
        *,
        device: Optional[Device] = None,
        dtype=None,
        requires_grad=True,
        **kwargs
    ):
        if isinstance(array, Tensor):
            if device is None:
                device = array.device
            if dtype is None:
                dtype = array.dtype
            
            if device == array.device and dtype == array.dtype:
                cached_data = array.realize_cached_data()
            else:
                # Needle cannot convert array data type in place. Cached data
                # are copied as and converted from `NDArray`.
                cached_data = Tensor._array_from_numpy(
                    array.numpy(), device=device, dtype=dtype
                )
        else:
            device = device if device else cpu()
            cached_data = Tensor._array_from_numpy(array, device=device, dtype=dtype)

        self._init(
            None,
            [],
            cached_data=cached_data,
            requires_grad=requires_grad,
        )
    # convert NumPy array to (customized) NDArray
    @staticmethod
    def _array_from_numpy(numpy_array, device, dtype):
        if array_api is numpy:
            return numpy.array(numpy_array, dtype=dtype)
        return array_api.array(numpy_array, device=device, dtype=dtype)

    @staticmethod
    def make_from_op(op: Op, inputs: List["Value"]):
        tensor = Tensor.__new__(Tensor)
        tensor._init(op, inputs)
        if not LAZY_MODE:
            if not tensor.requires_grad:
                return tensor.detach()
            tensor.realize_cached_data()
        return tensor

    @staticmethod
    def make_const(data, requires_grad=False):
        tensor = Tensor.__new__(Tensor)
        tensor._init(
            None,
            [],
            cached_data=data if   not isinstance(data, Tensor)
                             else data.realize_cached_data(),
            requires_grad=requires_grad,
        )
        return tensor
    # The `@property` decorator renders the method a getter.
    @property
    def data(self):
        return self.detach()
    # The `@….setter` decorator must follow the `@property` decorator.
    @data.setter
    def data(self, value):
        assert isinstance(value, Tensor)
        assert value.dtype == self.dtype, "%s %s" % (value.dtype,
                                                     self.dtype)
        self.cached_data = value.realize_cached_data()

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tensor.make_const(self.realize_cached_data())

    @property
    def shape(self):
        return self.realize_cached_data().shape # NDArray attribute

    @property
    def dtype(self):
        return self.realize_cached_data().dtype # NDArray attribute

    @property
    def device(self):
        data = self.realize_cached_data()
        # NumPy arrays always sit on CPUs.
        if array_api is numpy:
            return cpu()
        return data.device # NDArray attribute

    def backward(self, out_grad=None):
        out_grad = out_grad if out_grad \
                            else init.ones(*self.shape,
                                           dtype=self.dtype,
                                           device=self.device)
        compute_gradient_of_variables(self, out_grad)

    def __repr__(self):
        return "needle.Tensor(" + str(self.realize_cached_data()) + ")"

    def __str__(self):
        return self.realize_cached_data().__str__()

    def numpy(self):
        data = self.realize_cached_data()
        if array_api is numpy:
            return data
        return data.numpy()

    def __add__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, other)
        else:
            return needle.ops.AddScalar(other)(self)

    def __mul__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseMul()(self, other)
        else:
            return needle.ops.MulScalar(other)(self)

    def __pow__(self, other):
        ### BEGIN YOUR SOLUTION
        if isinstance(other, Tensor):
            raise NotImplementedError()
        else:
            return needle.ops.PowerScalar(other)(self)
        ### END YOUR SOLUTION

    def __neg__(self):
        return needle.ops.Negate()(self)

    def __sub__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseAdd()(self, needle.ops.Negate()(other))
        else:
            return needle.ops.AddScalar(-other)(self)

    def __truediv__(self, other):
        if isinstance(other, Tensor):
            return needle.ops.EWiseDiv()(self, other)
        else:
            return needle.ops.DivScalar(other)(self)
    # support the `@` operator
    def __matmul__(self, other):
        return needle.ops.MatMul()(self, other)
    # support the `matmul` method
    def matmul(self, other):
        return needle.ops.MatMul()(self, other)

    def sum(self, axes=None):
        return needle.ops.Summation(axes)(self)

    def broadcast_to(self, shape):
        return needle.ops.BroadcastTo(shape)(self)

    def reshape(self, shape):
        return needle.ops.Reshape(shape)(self)

    def transpose(self, axes=None):
        return needle.ops.Transpose(axes)(self)
    # support commutativity
    __radd__ = __add__
    __rsub__ = __sub__
    __rmul__ = __mul__
    __rmatmul__ = __matmul__


class TensorTuple(Value):
    """Represent a tuple of tensors.

    To keep things simple, we do not support nested tuples.
    """

    def __len__(self):
        cdata = self.realize_cached_data()
        return len(cdata)

    def __getitem__(self, index: int):
        return needle.ops.tuple_get_item(self, index)

    def tuple(self):
        return tuple([x for x in self])

    def __repr__(self):
        return "needle.TensorTuple" + str(self.tuple())

    def __str__(self):
        return self.__repr__()

    def __add__(self, other):
        assert isinstance(other, TensorTuple)
        assert len(self) == len(other)
        return needle.ops.make_tuple(*[self[i] + other[i] for i in range(len(self))])

    def detach(self):
        """Create a new tensor that shares the data but detaches from the graph."""
        return Tuple.make_const(self.realize_cached_data())


def compute_gradient_of_variables(output_tensor, out_grad):
    """Take gradient of output node with respect to each node in node_list.

    Store the computed result in the grad field of each Variable.
    """
    # a map from node to a list of gradient contributions from each output node
    node_to_output_grads_list: Dict[Tensor, List[Tensor]] = {}
    # Special note on initializing gradient of
    # We are really taking a derivative of the scalar reduce_sum(output_node)
    # instead of the vector output_node. But this is the common case for loss function.
    node_to_output_grads_list[output_tensor] = [out_grad]

    # Traverse graph in reverse topological order given the output_node that we are taking gradient wrt.
    reverse_topo_order = list(reversed(find_topo_sort([output_tensor])))

    ### BEGIN YOUR SOLUTION
    for node in reverse_topo_order:
        # get list of grad contributions
        output_grads_list = node_to_output_grads_list[node]
        # compute grad of current node w.r.t. output node
        # WARNING
        # The routine to sum a list of tensors is provided at the bottom. Do
        # not use Python built-in `sum`, which may lead to `dtype` mismatch.
        # `dtype` coherence is significant during optimazation.
        #
        # `hw1.ipynb` elaborates on this caveat.
        node.grad = sum_node_list(output_grads_list)
        # propagate grad to inputs
        if not node.is_leaf():
            # Typically, one should call `Op.gradient_as_tuple` to convert the
            # output of `….gradient` to a tuple so that it may be zipped with
            # `.inputs` as an iterable. I deliberately always return a tuple
            # in `ops`, thus, there is no need to call `gradient_as_tuple`.
            for node_input, grad in zip(node.inputs,
                                        node.op.gradient(node.grad, node)): # The gradient of an operation
                if node_input not in node_to_output_grads_list:             # is deliberately set as a
                    node_to_output_grads_list[node_input] = [grad]          # tuple even if there is a
                else:                                                       # single gradient.
                    node_to_output_grads_list[node_input].append(grad)
    ### END YOUR SOLUTION


def find_topo_sort(node_list: List[Value]) -> List[Value]:
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.
    """
    ### BEGIN YOUR SOLUTION
    # WARNING
    # 1. `node_list` contains only the output nodes.
    # 2. A computational graph is implemented in the opposite direction of
    #    mathematical operations in that `input` field (instead of the output)
    #    of a `Tensor` is defined. Hence, "post-order" means from "leaves" to
    #    loss.
    topo_order = []
    visited = set()
    for node in node_list:
        if id(node) not in visited:
            topo_sort_dfs(node, visited, topo_order)
    return topo_order
    ### END YOUR SOLUTION


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    ### BEGIN YOUR SOLUTION
    visited.add(id(node))

    if node.is_leaf():
        topo_order.append(node)
    else:
        for node_input in node.inputs:
            if id(node_input) not in visited:
                topo_sort_dfs(node_input, visited, topo_order)
        topo_order.append(node)
    ### END YOUR SOLUTION


##############################
####### Helper Methods #######
##############################


#def sum_node_list(node_list):
#    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
#    return reduce(add, node_list)
sum_node_list = lambda node_list : reduce(add, node_list)
