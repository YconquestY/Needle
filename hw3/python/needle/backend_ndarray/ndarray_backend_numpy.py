import numpy as np


__device_name__ = "numpy"
_datatype = np.float32 # single precision FP
_datetype_size = np.dtype(_datatype).itemsize


class Array:
    def __init__(self, size):
        self.array = np.empty(size, dtype=np.float32) # 1D array of single precision FP

    @property
    def size(self):
        # see https://numpy.org/doc/stable/reference/generated/numpy.ndarray.size.html
        return self.array.size


#def to_numpy(a: Array,
#             shape, strides, offset):
#    return np.lib.stride_tricks.as_strided(
#        a.array[offset:], shape, tuple([s * _datetype_size for s in strides])
#    )
to_numpy = lambda a, shape, strides, offset : \
                  np.lib.stride_tricks.as_strided(a.array[offset: ], shape,
                                                  (s * _datetype_size for s in strides))


def from_numpy(a: Array, out: Array):
    # see https://numpy.org/doc/stable/reference/generated/numpy.ndarray.flatten.html
    out.array[:] = a.flatten()


def fill(out: Array, val):
    # see https://numpy.org/doc/stable/reference/generated/numpy.ndarray.fill.html
    out.array.fill(val)


def compact(a: Array, out: Array,
            shape, strides, offset):
    out.array[:] = to_numpy(a, shape, strides, offset).flatten()


def ewise_setitem(a: Array, out: Array,
                  shape, strides, offset):
    to_numpy(out, shape, strides, offset)[:] = a.array.reshape(shape)


def scalar_setitem(size, val, out: Array,
                   shape, strides, offset):
    to_numpy(out, shape, strides, offset)[:] = val


def ewise_add(a: Array, b: Array, out: Array):
    out.array[:] = a.array + b.array


def scalar_add(a: Array, val, out: Array):
    out.array[:] = a.array + val


def ewise_mul(a: Array, b: Array, out: Array):
    out.array[:] = a.array * b.array


def scalar_mul(a: Array, val, out: Array):
    out.array[:] = a.array * val


def ewise_div(a: Array, b: Array, out: Array):
    out.array[:] = a.array / b.array


def scalar_div(a: Array, val, out: Array):
    out.array[:] = a.array / val


def scalar_power(a: Array, val, out: Array):
    out.array[:] = a.array ** val


def ewise_maximum(a: Array, b: Array, out: Array):
    out.array[:] = np.maximum(a.array, b.array)


def scalar_maximum(a: Array, val, out: Array):
    out.array[:] = np.maximum(a.array, val)


def ewise_eq(a: Array, b: Array, out: Array):
    out.array[:] = (a.array == b.array).astype(np.float32)


def scalar_eq(a: Array, val, out: Array):
    out.array[:] = (a.array == val).astype(np.float32)


def ewise_ge(a: Array, b: Array, out: Array):
    out.array[:] = (a.array >= b.array).astype(np.float32)


def scalar_ge(a: Array, val, out: Array):
    out.array[:] = (a.array >= val).astype(np.float32)


def ewise_log(a: Array, out: Array):
    out.array[:] = np.log(a.array)


def ewise_exp(a: Array, out: Array):
    out.array[:] = np.exp(a.array)


def ewise_tanh(a: Array, out: Array):
    out.array[:] = np.tanh(a.array)


def matmul(a: Array, b: Array, out: Array,
           m, n, p):
    out.array[:] = (a.array.reshape(m, n) @ b.array.reshape(n, p)).reshape(-1)


def reduce_max(a: Array, out: Array,
               reduce_size):
    out.array[:] = a.array[:].reshape(-1, reduce_size).max(axis=1)


def reduce_sum(a: Array, out: Array,
               reduce_size):
    out.array[:] = a.array[:].reshape(-1, reduce_size).sum(axis=1)
