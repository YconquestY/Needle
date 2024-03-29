#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <iostream>
#include <stdexcept>

namespace needle {
namespace cpu {

#define ALIGNMENT 256
#define TILE 8
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);


/**
 * This is a utility structure for maintaining an array aligned to ALIGNMENT
 * boundaries in memory. This alignment should be at least TILE * ELEM_SIZE,
 * though we make it even larger here by default.
 */
struct AlignedArray
{
    AlignedArray(const size_t size)
    {
        // see https://pubs.opengroup.org/onlinepubs/9699919799/functions/posix_memalign.html
        // Weirdly, there is no manual for the API from CL.
        int ret = posix_memalign((void**) &ptr, ALIGNMENT, size * ELEM_SIZE);
        if (ret != 0) {
            throw std::bad_alloc();
        }
        this->size = size;
    }
    ~AlignedArray() {
        free(ptr);
    }
    size_t ptr_as_int() {
        return (size_t) ptr;
    }
    scalar_t *ptr;
    size_t   size;
};


void Fill(AlignedArray* out, scalar_t val)
{
    /**
     * Fill the an aligned array with `val`
     */
    for (int i = 0; i < out->size; i++) {
        out->ptr[i] = val;
    }
}

// Use signed integers when dealing with `strides`. Otherwise, flipping an
// NDArray will fail.
// see https://forum.dlsyscourse.org/t/q3-flip-typeerror-to-numpy-incompatible-function-arguments/2862/9

/// BEGIN YOUR SOLUTION
uint32_t GetMemIdx(uint32_t idx, uint32_t* carry,
                   std::vector<uint32_t> shape, std::vector<int32_t> strides, size_t offset)
{
    /**
     * Get the actual index of an array element in memory. The index strategy
     * is implemented in a modular fashion so that it is used by `Compact` and
     * `…Setitem`.
     * 
     * Args:
     *   idx[uint32_t]   : index of the element in the (non-compact) array
     *   carry[uint32_t*]: a pointer to the carry array
     *   shape[vector<uint32_t>]  : shape   of the array
     *   strides[vector<uint32_t>]: strides of the array
     *   offset[size_t]           : offset  of the array
     * 
     * Returns:
     *   uint32_t: index of the element in memory
     */
    uint32_t mem_idx = offset;
    for (unsigned int i = 0; i < shape.size(); i++) {
        mem_idx += (idx / carry[i]) % shape[i] * strides[i];
    }
    return mem_idx;
}
/// END YOUR SOLUTION


// `a` is a reference rather than a pointer.
// This is because we do not write to a populated array.
// see https://stackoverflow.com/questions/57483/what-are-the-differences-between-a-pointer-variable-and-a-reference-variable
void Compact(const AlignedArray& a, AlignedArray* out,
             std::vector<uint32_t> shape, std::vector<int32_t> strides, size_t offset)
{
    /**
     * Compact an array in memory
     *
     * Args:
     *   a:       non-compact representation of the array, given as input
     *   out:     compact version of the array to be written
     *   shape:   shapes of each dimension for `a` and `out`
     *   strides: strides of the `a` array (not `out`, which has compact strides)
     *   offset:  offset  of the `a` array (not `out`, which has zero offset, being compact)
     *
     * Returns:
     *   void (you need to modify out directly, rather than returning anything; this is true for all the
     *   function will implement here, so we won't repeat this note.)
     */
    /// BEGIN YOUR SOLUTION
    // compute total size
    // see https://en.cppreference.com/w/cpp/algorithm/accumulate
    //     https://en.cppreference.com/w/cpp/utility/functional/multiplies
    uint32_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>());
    out->size = size; // necessary?
    // determine carry "boundaries"
    uint32_t *carry = new uint32_t[shape.size()];
    carry[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; i--) { // Do not use `unsigned int`.
        carry[i] = carry[i + 1] * shape[i + 1];
    }

    for (uint32_t idx = 0; idx < out->size; idx++) {
        out->ptr[idx] = a.ptr[GetMemIdx(idx, carry, shape, strides, offset)];
    }
    // free memory
    delete[] carry;
    /// END YOUR SOLUTION
}

void EwiseSetitem(const AlignedArray& a, AlignedArray* out,
                  std::vector<uint32_t> shape, std::vector<int32_t> strides, size_t offset)
{
    /**
     * Set items in a (non-compact) array
     *
     * Args:
     *   a:       compact array whose items will be written to `out`
     *   out:     non-compact array whose items are to be written
     *   shape:   shapes of each dimension for `a` and `out`
     *   strides: strides of the `out` array (not `a`, which has compact strides)
     *   offset:  offset  of the `out` array (not `a`, which has zero offset, being compact)
     */
    /// BEGIN YOUR SOLUTION
    // compute total size
    uint32_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<uint32_t>());
    // determine carry "boundaries"
    uint32_t *carry = new uint32_t[shape.size()];
    carry[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; i--) { // Do not use `unsigned int`.
        carry[i] = carry[i + 1] * shape[i + 1];
    }

    for (uint32_t idx = 0; idx < a.size; idx++) {
        out->ptr[GetMemIdx(idx, carry, shape, strides, offset)] = a.ptr[idx];
    }
    // free memory
    delete[] carry;
    /// END YOUR SOLUTION
}

void ScalarSetitem(const size_t size, scalar_t val, AlignedArray* out,
                   std::vector<uint32_t> shape, std::vector<int32_t> strides, size_t offset)
{
    /**
     * Set items in a (non-compact) array
     *
     * Args:
     *   size   : number of elements to write in `out` array (note that this is
     *            equal to `out.size`, because out is a non-compact subset
     *            array);  it is also the same as the product of items in
     *            `shape`, but convenient to just pass it here.
     *   val    : scalar value to write to
     *   out    : non-compact array whose items are to be written
     *   shape  : shapes  of each dimension of out
     *   strides: strides of the out array
     *   offset : offset  of the out array
     */

    /// BEGIN YOUR SOLUTION
    // determine carry "boundaries"
    uint32_t *carry = new uint32_t[shape.size()];
    carry[shape.size() - 1] = 1;
    for (int i = shape.size() - 2; i >= 0; i--) { // Do not use `unsigned int`.
        carry[i] = carry[i + 1] * shape[i + 1];
    }

    for (uint32_t idx = 0; idx < size; idx++) {
        out->ptr[GetMemIdx(idx, carry, shape, strides, offset)] = val;
    }
    // free memory
    delete[] carry;
    /// END YOUR SOLUTION
}


/**
 * In the code that follows, use the above template to create analogous
 * element-wise and scalar operators for the following functions. See the NumPy
 * backend for examples on how they should work.
 * 
 *   - EwiseMul, ScalarMul
 *   - EwiseDiv, ScalarDiv
 *   - ScalarPower
 *   - EwiseMaximum, ScalarMaximum
 *   - EwiseEq, ScalarEq
 *   - EwiseGe, ScalarGe
 *   - EwiseLog
 *   - EwiseExp
 *   - EwiseTanh
 *
 * If you implement all these naively, there will be a lot of repeated code, so
 * you are welcome (but not required) to use macros or templates to define
 * these functions (however you want to do so, as long as the functions match
 * the proper) signatures above.
 */

/// BEGIN YOUR SOLUTION
void EwiseOp(const AlignedArray& a, const AlignedArray& b, AlignedArray* out,
             scalar_t (*op)(scalar_t, scalar_t)) // function/template as parameter
{
    /**
     * set entries in `out` to be the result of applying `op` to corresponding
     * entries in `a` and `b`
     */
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = op(a.ptr[i], b.ptr[i]);
    }
}
// overloaded function
void EwiseOp(const AlignedArray& a, AlignedArray* out,
             scalar_t (*op)(scalar_t))
{
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = op(a.ptr[i]);
    }
}

void ScalarOp(const AlignedArray& a, scalar_t val, AlignedArray* out,
              scalar_t (*op)(scalar_t, scalar_t))
{
    /**
     * set entries in `out` to be the result of applying `op` to corresponding
     * entry in `a` and `val`
     */
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = op(a.ptr[i], val);
    }
}
/*
void PScalarOp(const AlignedArray& a, const AlignedArray& b, AlignedArray* out,
               scalar_t (*op)(scalar_t, scalar_t))
{
    for (size_t i = 0; i < a.size; i++) {
        out->ptr[i] = op(a.ptr[i], b.ptr[0]);
    }
}
*/
scalar_t add(scalar_t a, scalar_t b) { return a + b; }
scalar_t mul(scalar_t a, scalar_t b) { return a * b; }
scalar_t div(scalar_t a, scalar_t b) { return a / b; }
// std::pow
scalar_t max(scalar_t a, scalar_t b) { return a > b ? a : b; }
float eq(scalar_t a, scalar_t b) { return a == b ? 1.0f : 0.0f; }
float ge(scalar_t a, scalar_t b) { return a >= b ? 1.0f : 0.0f; }
// std::log
// std::exp
// std::tanh

void EwiseAdd    (const AlignedArray& a, const AlignedArray& b, AlignedArray* out) { EwiseOp(a, b, out, add); }
void EwiseMul    (const AlignedArray& a, const AlignedArray& b, AlignedArray* out) { EwiseOp(a, b, out, mul); }
void EwiseDiv    (const AlignedArray& a, const AlignedArray& b, AlignedArray* out) { EwiseOp(a, b, out, div); }
void EwiseMaximum(const AlignedArray& a, const AlignedArray& b, AlignedArray* out) { EwiseOp(a, b, out, max); }
void EwiseEq     (const AlignedArray& a, const AlignedArray& b, AlignedArray* out) { EwiseOp(a, b, out, eq ); }
void EwiseGe     (const AlignedArray& a, const AlignedArray& b, AlignedArray* out) { EwiseOp(a, b, out, ge ); }
void EwiseLog    (const AlignedArray& a,                        AlignedArray* out) { EwiseOp(a,    out, std::log ); }
void EwiseExp    (const AlignedArray& a,                        AlignedArray* out) { EwiseOp(a,    out, std::exp ); }
void EwiseTanh   (const AlignedArray& a,                        AlignedArray* out) { EwiseOp(a,    out, std::tanh); }

void ScalarAdd    (const AlignedArray& a, scalar_t val, AlignedArray* out) { ScalarOp(a, val, out, add); }
void ScalarMul    (const AlignedArray& a, scalar_t val, AlignedArray* out) { ScalarOp(a, val, out, mul); }
void ScalarDiv    (const AlignedArray& a, scalar_t val, AlignedArray* out) { ScalarOp(a, val, out, div); }
void ScalarPower  (const AlignedArray& a, scalar_t val, AlignedArray* out) { ScalarOp(a, val, out, std::pow); }
void ScalarMaximum(const AlignedArray& a, scalar_t val, AlignedArray* out) { ScalarOp(a, val, out, max); }
void ScalarEq     (const AlignedArray& a, scalar_t val, AlignedArray* out) { ScalarOp(a, val, out, eq ); }
void ScalarGe     (const AlignedArray& a, scalar_t val, AlignedArray* out) { ScalarOp(a, val, out, ge ); }

//void PScalarAdd(const AlignedArray& a, AlignedArray& b, AlignedArray* out) { PScalarOp(a, b, out, add); }
/// END YOUR SOLUTION

void Matmul(const AlignedArray& a, const AlignedArray& b, AlignedArray* out,
            uint32_t m, uint32_t n, uint32_t p)
{
    /**
     * Multiply two (compact) matrices into an output (also compact) matrix.  For this implementation
     * you can use the "naive" three-loop algorithm.
     *
     * Args:
     *   a:   compact 2D array of size m x n
     *   b:   compact 2D array of size n x p
     *   out: compact 2D array of size m x p to write the output to
     *   m: rows    of a / out
     *   n: columns of a / rows of b
     *   p: columns of b / out
     */

    /// BEGIN YOUR SOLUTION
    for (uint32_t i = 0; i < m; i++)
    {
        for (uint32_t j = 0; j < p; j++)
        {
            scalar_t entry = 0.0f;
            for (uint32_t k = 0; k < n; k++) {
                entry += a.ptr[i * n + k] * b.ptr[k * p + j];
            }
            out->ptr[i * p + j] = entry;
        }
    }
    /// END YOUR SOLUTION
}

inline void AlignedDot(const float* __restrict__ a,
                       const float* __restrict__ b,
                             float* __restrict__ out)
{
    /**
     * Multiply together two TILE x TILE matrices, and add the result to `out`
     * (it is important to add the result to the existing out, which you should
     * not set to zero beforehand). We are including the compiler flags here
     * that enable the compile to properly use vector operators to implement
     * this function. Specifically, the `__restrict__` keyword indicates to the
     * compiler that `a`, `b`, and `out` don't have any overlapping memory,
     * which is necessary in order for vector operations to be equivalent to
     * their non-vectorized counterparts (imagine what could happen otherwise
     * if `a`, `b`, and `out` had overlapping memory). Similarly, the
     * `__builtin_assume_aligned` keyword tells the compiler that the input
     * array will be aligned to the appropriate blocks in memory, which also
     * helps the compiler vectorize the code.
     *
     * Args:
     *   a:   compact 2D array of size TILE x TILE
     *   b:   compact 2D array of size TILE x TILE
     *   out: compact 2D array of size TILE x TILE to write to
     */

    a   = (const float*) __builtin_assume_aligned(a  , TILE * ELEM_SIZE);
    b   = (const float*) __builtin_assume_aligned(b  , TILE * ELEM_SIZE);
    out = (      float*) __builtin_assume_aligned(out, TILE * ELEM_SIZE);

    /// BEGIN YOUR SOLUTION
    for (uint32_t i = 0; i < TILE; i++)
    {
        for (uint32_t j = 0; j < TILE; j++)
        {
            scalar_t entry = 0.0f;
            for (uint32_t k = 0; k < TILE; k++) {
                entry += a[i * TILE + k] * b[k * TILE + j];
            }
            out[i * TILE + j] += entry;
        }
    }
    /// END YOUR SOLUTION
}

void MatmulTiled(const AlignedArray& a, const AlignedArray& b, AlignedArray* out,
                 uint32_t m, uint32_t n, uint32_t p)
{
    /**
     * Matrix multiplication on tiled representations of array.  In this setting, a, b, and out
     * are all 4D compact arrays of the appropriate size, e.g. a is an array of size
     *   a[m/TILE][n/TILE][TILE][TILE]
     * You should do the multiplication tile-by-tile to improve performance of the array (i.e., this
     * function should call `AlignedDot()` implemented above).
     *
     * Note that this function will only be called when m, n, p are all multiples of TILE, so you can
     * assume that this division happens without any remainder.
     *
     * Args:
     *   a:   compact 4D array of size m/TILE x n/TILE x TILE x TILE
     *   b:   compact 4D array of size n/TILE x p/TILE x TILE x TILE
     *   out: compact 4D array of size m/TILE x p/TILE x TILE x TILE to write to
     *   m: rows    of a / out
     *   n: columns of a / rows of b
     *   p: columns of b / out
     *
     */
    /// BEGIN YOUR SOLUTION
    // Tiled MM requires accumulating intermediate results. Thus, `out->ptr`
    // must be initialized to zeros. There are cases, in practice, where
    // initial elements are extremely large, which means we have to zero out
    // the array manually.
    // see https://en.cppreference.com/w/cpp/string/byte/memset
    out->ptr = (scalar_t*) std::memset((void*) out->ptr, 0, m * p * sizeof(scalar_t));

    uint32_t m_tiles = (m + TILE - 1) / TILE,
             n_tiles = (n + TILE - 1) / TILE,
             p_tiles = (p + TILE - 1) / TILE;
    for (uint32_t i = 0; i < m_tiles; i++)
    {
        for (uint32_t j = 0; j < p_tiles; j++)
        {
            for (uint32_t k = 0; k < n_tiles; k++)
            {
                AlignedDot(a.ptr    + (i * n_tiles + k) * TILE * TILE,
                           b.ptr    + (k * p_tiles + j) * TILE * TILE,
                           out->ptr + (i * p_tiles + j) * TILE * TILE);
            }
        }
    }
    /// END YOUR SOLUTION
}

void ReduceMax(const AlignedArray& a, AlignedArray* out, size_t reduce_size)
{
    /**
     * Reduce by taking maximum over `reduce_size` contiguous blocks.
     *
     * Args:
     *   a  : compact array of size a.size = out.size x reduce_size to reduce over
     *   out: compact array to write into
     *   reduce_size: size of the dimension to reduce over
     */

    /// BEGIN YOUR SOLUTION
    for (size_t i = 0; i < out->size; i++) {
        // `std::max_element` returns a pointer (iterator) rather than the
        // value itself,
        // see https://en.cppreference.com/w/cpp/algorithm/max_element
        out->ptr[i] = *std::max_element(a.ptr + i * reduce_size,
                                        a.ptr + (i + 1) * reduce_size);
    }
    /// END YOUR SOLUTION
}

void ReduceSum(const AlignedArray& a, AlignedArray* out, size_t reduce_size)
{
    /**
     * Reduce by taking sum over `reduce_size` contiguous blocks.
     *
     * Args:
     *   a  : compact array of size a.size = out.size x reduce_size to reduce over
     *   out: compact array to write into
     *   reduce_size: size of the dimension to reduce over
     */

    /// BEGIN YOUR SOLUTION
    for (size_t i = 0; i < out->size; i++) {
        out->ptr[i] = std::accumulate(a.ptr + i * reduce_size,
                                      a.ptr + (i + 1) * reduce_size, 0.0); // 0.0 instead of 0!
    }
    /// END YOUR SOLUTION
}


}  // namespace cpu
}  // namespace needle

PYBIND11_MODULE(ndarray_backend_cpu, m) {
  namespace py = pybind11;
  using namespace needle;
  using namespace cpu;

  m.attr("__device_name__") = "cpu";
  m.attr("__tile_size__") = TILE;

  py::class_<AlignedArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def("ptr", &AlignedArray::ptr_as_int)
      .def_readonly("size", &AlignedArray::size);

  // return numpy array (with copying for simplicity, otherwise garbage
  // collection is a pain)
  m.def("to_numpy", [](const AlignedArray& a, std::vector<size_t> shape,
                       std::vector<size_t> strides, size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });
    return py::array_t<scalar_t>(shape, numpy_strides, a.ptr + offset);
  });

  // convert from numpy (with copying)
  m.def("from_numpy", [](py::array_t<scalar_t> a, AlignedArray* out) {
    std::memcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE);
  });

    m.def("fill"   , Fill   );
    m.def("compact", Compact);
    m.def("ewise_setitem" , EwiseSetitem );
    m.def("scalar_setitem", ScalarSetitem);

    m.def("ewise_add"  , EwiseAdd  );
    m.def("scalar_add" , ScalarAdd );
    //m.def("pscalar_add", PScalarAdd);
    m.def("ewise_mul"  , EwiseMul  );
    m.def("scalar_mul" , ScalarMul );
    m.def("ewise_div"  , EwiseDiv  );
    m.def("scalar_div" , ScalarDiv );
    m.def("scalar_power", ScalarPower);

    m.def("ewise_maximum" , EwiseMaximum );
    m.def("scalar_maximum", ScalarMaximum);
    m.def("ewise_eq" , EwiseEq );
    m.def("scalar_eq", ScalarEq);
    m.def("ewise_ge" , EwiseGe );
    m.def("scalar_ge", ScalarGe);

    m.def("ewise_log" , EwiseLog );
    m.def("ewise_exp" , EwiseExp );
    m.def("ewise_tanh", EwiseTanh);

    m.def("matmul", Matmul);
    m.def("matmul_tiled", MatmulTiled);

    m.def("reduce_max", ReduceMax);
    m.def("reduce_sum", ReduceSum);
}
