# [CMU 10-414/714](https://dlsyscourse.org)

This repository is for homework in [CMU 10-414/714](https://dlsyscourse.org) *Deep Learning Systems* (fall 2022).

## WARNING

- Needle does **not** support implicit broadcasting.
- Make sure to compact a tensor before reshaping it.

## TODO

- [ ] Extracting a scalar from an NDArray or a tensor<br>
  Needle does **not** support turning a single-entry tensor to a Python number via the [`item`](https://pytorch.org/docs/stable/generated/torch.Tensor.item.html) method. Nor does its backend array support "extracting" a Python scalar through indexing with [`__getitem__`](https://github.com/YconquestY/Needle/blob/hw/hw3/python/needle/backend_ndarray/ndarray.py#L333). This affects the choice of kernels when performing operations on singleton tensors. As NDArray, they are not recognized as scalars, and are **broadcast** before **elementwise** kernels are invoked. The broadcasting procedure wastes time as well as memory, and **a wiser solution is to be found.**

## (TL; DR) Needle explained

- Taking tensor addition for example, Needle operations are performed as follows:[^call_stack]<br>
  1. call `a + b` to add tensors `a` and `b`;
  2. `+` translates to `Tensor.__add__(a, b)`;
  3. invoke `EWiseAdd()(a, b)`;
  4. invoke `EwiseAdd().__call__(a, b)`, which inherits `TensorOp.__call__(a, b)`;
  5. invoke `Tensor.make_from_op((a, b))`<br>
     Note that inputs `a` and `b` are packed into a `tuple`.
  6. invoke `Tensor.__new__` and `Tensor._init` (inherited from `Value._init`) to construct the result tensor
  7. compute the result **immediately** if 1) in `LAZY_MODE` or 2) the tensor does not carries a gradient
## Hints

- `__repr__` (called by `repr`) and `__str__` (called by `str` or `print`) customize the string representation of an object for inspection.[^repr_str]
- "If a class defines a `__call__` method, then its instances may be invoked as functions."[^call]
- "A class runs its `__new__` method to create an instance."[^new]<br>
  `__init__`, which does not return anything, gets `self` as the first argument. An object already exists when `__init__` is called by the interpreter. Hence, it is an **initializer** instead of a constructor.[^init]
- "The `@property` decorator marks the **getter** method of a class attribute."[^property]
- **Following** `@property`, the `@….setter` decorator ties the getter and setter.

## References

[^call_stack]: [How do `autograd.py` and `ops.py` interact with each other?](https://forum.dlsyscourse.org/t/how-do-autograd-py-and-ops-py-interact-with-each-other/2435/3?u=will)

[^repr_str]: [What is the difference between `__str__` and `__repr__`?](https://stackoverflow.com/a/2626364)

[^call]: page $238$ of [*Fluent Python*](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/)

[^new]: page $238$ of [Fluent Python](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/)

[^init]: page $843$ of [Fluent Python](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/)

[^property]: page $375$ of [Fluent Python](https://www.oreilly.com/library/view/fluent-python-2nd/9781492056348/)
