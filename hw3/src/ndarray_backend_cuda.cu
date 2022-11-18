#include <cuda_runtime.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <sstream>
#include <cmath> // for `std::abs`

namespace needle {
namespace cuda {

#define BASE_THREAD_NUM 256

#define TILE 4
typedef float scalar_t;
const size_t ELEM_SIZE = sizeof(scalar_t);

struct CudaArray
{
    CudaArray(const size_t size)
    {
        cudaError_t err = cudaMalloc(&ptr, size * ELEM_SIZE);
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
        this->size = size;
    }
    ~CudaArray() {
        cudaFree(ptr);
    }
    size_t ptr_as_int() {
        return (size_t) ptr;
    }
    scalar_t* ptr;
    size_t    size;
};

struct CudaDims {
    dim3 block, grid;
};

CudaDims CudaOneDim(size_t size)
{
    /**
     * Utility function to get cuda dimensions for 1D call
     */
    CudaDims dim;
    size_t num_blocks = (size + BASE_THREAD_NUM - 1) / BASE_THREAD_NUM; // round up
    // 1D threads in 1D blocks
    dim.grid  = dim3(num_blocks     , 1, 1);
    dim.block = dim3(BASE_THREAD_NUM, 1, 1);

    return dim;
}

#define MAX_VEC_SIZE 8
struct CudaVec
{
    uint32_t size;
    uint32_t data[MAX_VEC_SIZE];
};

CudaVec VecToCuda(const std::vector<uint32_t>& x)
{
    CudaVec shape;
    // `shape` or `strides` cannot exceed 8D.
    if (x.size() > MAX_VEC_SIZE) {
        throw std::runtime_error("Exceeded CUDA supported max dimesions");
    }
    shape.size = x.size();
    for (size_t i = 0; i < x.size(); i++) {
        shape.data[i] = x[i];
    }
    return shape;
}

////////////////////////////////////////////////////////////////////////////////
// Fill call
////////////////////////////////////////////////////////////////////////////////

__global__ void FillKernel(scalar_t* out, scalar_t val, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        out[tid] = val;
    }
}

void Fill(CudaArray* out, scalar_t val)
{
    CudaDims dim = CudaOneDim(out->size);
    FillKernel<<<dim.grid, dim.block>>>(out->ptr, val, out->size);
}

////////////////////////////////////////////////////////////////////////////////
// Compact and setitem cals
////////////////////////////////////////////////////////////////////////////////

CudaVec GetCarry(std::vector<uint32_t> shape)
{
    /**
     * Get the carry boundaries for array indices
     */
    CudaVec carry;
    // Altough a tensor should not exceed 8D, we do not check it here. Instead,
    // it is handled by `VecToCuda(shape)`.
    carry.size = shape.size();

    carry.data[carry.size - 1] = 1;
    for (int i = carry.size - 2; i >= 0; i--) {
        carry.data[i] = carry.data[i + 1] * shape[i + 1];
    }

    return carry;
}


// Untility function to convert contiguous index i to memory location from strides
//
// The `__device__` execution space specifier declares a function callable only
// from a device, see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#device-function-specifier
__device__ uint32_t GetMemIdx(uint32_t tid, CudaVec carry,
                              CudaVec shape, CudaVec strides, size_t offset)
{
    /**
     * Get the actual index of an array element in memory. The index strategy
     * is implemented in a modular fashion so that it is used by `Compact` and
     * `…Setitem`.
     * 
     * Args:
     *   idx[uint32_t]   : index of the element in the (non-compact) array
     *   carry[uint32_t*]: a pointer to the carry array
     *   shape[CudaVec]  : shape   of the array
     *   strides[CudaVec]: strides of the array
     *   offset[size_t]  : offset  of the array
     * 
     * Returns:
     *   uint32_t: index of the element in memory
     */
    uint32_t mem_idx = offset;
    for (unsigned int i = 0; i < shape.size; i++) {
        mem_idx += (tid / carry.data[i]) % shape.data[i] * strides.data[i];
    }
    return mem_idx;
}


__global__ void CompactKernel(const scalar_t* a, scalar_t* out,
                              size_t size, CudaVec shape, CudaVec strides, size_t offset,
                              CudaVec carry)
{
    /**
     * The CUDA kernel for the compact opeation. This should effectively map a
     * single entry in the non-compact input `a`, to the corresponding item (at
     * location `tid`) in the compact array `out`.
     * 
     * Args:
     *   a  : CUDA pointer to array `a`
     *   out: CUDA pointer to array `out`
     *   size   : size   of `out` array
     *   shape  : vector of shapes of `a` and `out` arrays
     *            (of type CudaVec, for past passing to CUDA kernel)
     *   strides: vector of strides of out array
     *   offset : offset of `out` array
     */
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;

    /// BEGIN YOUR SOLUTION
    if (tid < size) {
        out[tid] = a[GetMemIdx(tid, carry, shape, strides, offset)];
    }
    /// END YOUR SOLUTION
}

void Compact(const CudaArray& a, CudaArray* out,
             std::vector<uint32_t> shape, std::vector<uint32_t> strides, size_t offset)
{
    /**
     * Compact an array in memory. Unlike the C++ version, in CUDA this will
     * primarily call the relevant CUDA kernel. In this case, we illustrate how
     * you should set this up (i.e., we give you the code for this fuction, and
     * also the prototype for the CompactKernel() function). For the functions
     * after this, however, you'll need to define these kernels as you see fit
     * to execute the underlying function.
     * 
     * Args:
     *   a  : non-compact represntation of the array, given as input
     *   out: compact version of the array to be written
     *   shape  : shapes  of each dimension for a and out
     *   strides: strides of the `a` array (not `out`, which has compact strides)
     *   offset : offset  of the `a` array (not `out`, which has zero offset, being compact)
     */

    // Despite the claim "nothing needs to be added here", I insert an argument
    // to kernel invocation.
    CudaDims dim = CudaOneDim(out->size);
    CompactKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr,
                                           out->size, VecToCuda(shape), VecToCuda(strides), offset,
                                           GetCarry(shape));
}


/// BEGIN YOUR SOLUTION
__global__ void EwiseSetitemKernel(const scalar_t* a, scalar_t* out,
                                   size_t size, CudaVec shape, CudaVec strides, size_t offset,
                                   CudaVec carry)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        out[GetMemIdx(tid, carry, shape, strides, offset)] = a[tid];
    }
}
/// END YOUR SOLUTION

void EwiseSetitem(const CudaArray& a, CudaArray* out,
                  std::vector<uint32_t> shape, std::vector<uint32_t> strides, size_t offset)
{
    /**
     * Set items in a (non-compact) array using CUDA.  You will most likely want to implement a
     * EwiseSetitemKernel() function, similar to those above, that will do the actual work.
     * 
     * Args:
     *   a: _compact_ array whose items will be written to out
     *   out: non-compact array whose items are to be written
     *   shape: shapes of each dimension for a and out
     *   strides: strides of the *out* array (not a, which has compact strides)
     *   offset: offset of the *out* array (not a, which has zero offset, being compact)
     */
    /// BEGIN YOUR SOLUTION
    // Use `a.size` when creating `CudaDims` bacause `a` is compact.
    // see https://forum.dlsyscourse.org/t/q6-setitem-in-gpu-version/2641/3
    CudaDims dim = CudaOneDim(a.size);
    EwiseSetitemKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr,
                                                a.size, VecToCuda(shape), VecToCuda(strides), offset,
                                                GetCarry(shape));
    /// END YOUR SOLUTION
}


/// BEGIN YOUR SOLUTION
__global__ void ScalarSetitemKernel(scalar_t val, scalar_t* out,
                                    size_t size, CudaVec shape, CudaVec strides, size_t offset,
                                    CudaVec carry)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        out[GetMemIdx(tid, carry, shape, strides, offset)] = val;
    }
}
/// END YOUR SOLUTION

void ScalarSetitem(size_t size, scalar_t val, CudaArray* out,
                   std::vector<uint32_t> shape, std::vector<uint32_t> strides, size_t offset)
{
    /**
     * Set items is a (non-compact) array
     * 
     * Args:
     *   size: number of elements to write in out array (note that this will note be the same as
     *         out.size, because out is a non-compact subset array);  it _will_ be the same as the 
     *         product of items in shape, but covenient to just pass it here.
     *   val: scalar value to write to
     *   out: non-compact array whose items are to be written
     *   shape: shapes of each dimension of out
     *   strides: strides of the out array
     *   offset: offset of the out array
     */
    /// BEGIN YOUR SOLUTION
    // Do not use `out->size` when creating `CudaDims` bacause `out` is not compact.
    // see https://forum.dlsyscourse.org/t/q6-setitem-in-gpu-version/2641/3
    CudaDims dim = CudaOneDim(size);
    ScalarSetitemKernel<<<dim.grid, dim.block>>>(val, out->ptr,
                                                 size, VecToCuda(shape), VecToCuda(strides), offset,
                                                 GetCarry(shape));
    /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void EwiseAddKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) out[tid] = a[tid] + b[tid];
}

void EwiseAdd(const CudaArray& a, const CudaArray& b, CudaArray* out) {
    /**
     * Add together two CUDA array
     */
    CudaDims dim = CudaOneDim(out->size);
    EwiseAddKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size);
}

__global__ void ScalarAddKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) out[tid] = a[tid] + val;
}

void ScalarAdd(const CudaArray& a, scalar_t val, CudaArray* out) {
    /**
     * Add together a CUDA array and a scalar value.
     */
    CudaDims dim = CudaOneDim(out->size);
    ScalarAddKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size);
}

/**
 * In the code the follows, use the above template to create analogous elementise
 * and and scalar operators for the following functions.  See the NumPy backend
 * for examples of how they should work.
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
 * you are welcome (but not required), to use macros or templates to define these
 * functions (however you want to do so, as long as the functions match the proper)
 * signatures above.
 */

/// BEGIN YOUR SOLUTION
#define EPS 1e-6 // tolerance of `float` comparison
// CUDA does not support passing STL templates to kernels.
// see https://stackoverflow.com/questions/14874351/cuda-kernel-launch-macro-with-templates
#define EWISE_BIN_OP(a, b, out,                       \
                     OpKernel)                        \
            do {                                      \
                CudaDims dim = CudaOneDim(out->size); \
                OpKernel<<<dim.grid, dim.block>>>(a.ptr, b.ptr, out->ptr, out->size); \
            } while (0)
// overloaded micro
#define EWISE_OP(a, out,                              \
                 OpKernel)                            \
            do {                                      \
                CudaDims dim = CudaOneDim(out->size); \
                OpKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, out->size); \
            } while (0)

#define SCALAR_OP(a, val, out,                        \
                 OpKernel)                            \
            do {                                      \
                CudaDims dim = CudaOneDim(out->size); \
                OpKernel<<<dim.grid, dim.block>>>(a.ptr, val, out->ptr, out->size); \
            } while (0)


__global__ void EwiseMulKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        out[tid] = a[tid] * b[tid];
    }
}

__global__ void EwiseDivKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        out[tid] = a[tid] / b[tid];
    }
}

__global__ void EwiseMaximumKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // see https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g6e7516db46be25c33fb26e203287f2a3
        out[tid] = fmaxf(a[tid], b[tid]);
    }
}

__global__ void EwiseEqKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // see https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1gb00f8593e1bfb1985526020fbec4e0fc
        out[tid] = (fabsf(a[tid] - b[tid]) < EPS ? 1.0f : 0.0f);
    }
}

__global__ void EwiseGeKernel(const scalar_t* a, const scalar_t* b, scalar_t* out, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        out[tid] = (a[tid] >= b[tid] ? 1.0f : 0.0f);
    }
}

__global__ void EwiseLogKernel(const scalar_t* a, scalar_t* out, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // see https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__SINGLE.html#group__CUDA__MATH__INTRINSIC__SINGLE_1ged5cef656578096892f104a27d5287c4
        out[tid] = __logf(a[tid]);
    }
}

__global__ void EwiseExpKernel(const scalar_t* a, scalar_t* out, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // see https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__SINGLE.html#group__CUDA__MATH__INTRINSIC__SINGLE_1g1beeb3ae544cfdde4a0a724ace025aed
        out[tid] = __expf(a[tid]);
    }
}

__global__ void EwiseTanhKernel(const scalar_t* a, scalar_t* out, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // see https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g7d925743801795775ca98ae83d4ba6e6
        out[tid] = tanhf(a[tid]);
    }
}


__global__ void ScalarMulKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) out[tid] = a[tid] * val;
}

__global__ void ScalarDivKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) out[tid] = a[tid] / val;
}

// overloaded kernels for `ScalarPower`
// see https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#math-libraries
__global__ void InvKernel(const scalar_t* a, scalar_t* out, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // see https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__SINGLE.html#group__CUDA__MATH__INTRINSIC__SINGLE_1gba455801af8ac9af405a5d37ef2f077b
        out[tid] = __frcp_rn(a[tid]);
    }
}

__global__ void InvSqrtKernel(const scalar_t* a, scalar_t* out, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // see https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__SINGLE.html#group__CUDA__MATH__INTRINSIC__SINGLE_1g71ee45580cbeeea206297f0112aff42c
        out[tid] = __frsqrt_rn(a[tid]);
    }
}

__global__ void InvCbrtKernel(const scalar_t* a, scalar_t* out, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // see https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g96d2384128af36ea9cb9b20d366900c7
        out[tid] = rcbrtf(a[tid]);
    }
}

__global__ void IdentityKernel(const scalar_t* a, scalar_t* out, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        out[tid] = 1.0f;
    }
}

__global__ void CbrtKernel(const scalar_t* a, scalar_t* out, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // see https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__SINGLE.html#group__CUDA__MATH__SINGLE_1g96d2384128af36ea9cb9b20d366900c7
        out[tid] = cbrtf(a[tid]);
    }
}

__global__ void SqrtKernel(const scalar_t* a, scalar_t* out, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // see https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__SINGLE.html#group__CUDA__MATH__INTRINSIC__SINGLE_1gf021e85b5e9de141a0fc2ff6fbe85875
        out[tid] = __fsqrt_rn(a[tid]);
    }
}

__global__ void CopyKernel(const scalar_t* a, scalar_t* out, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        out[tid] = a[tid];
    }
}
// "For small integer powers (e.g., x² or x³), explicit multiplication is
// almost certainly faster than the use of general exponentiation routines such
// as `pow()`…"
__global__ void SquareKernel(const scalar_t* a, scalar_t* out, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        out[tid] = a[tid] * a[tid];
    }
}

__global__ void CubeKernel(const scalar_t* a, scalar_t* out, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        out[tid] = a[tid] * a[tid] * a[tid];
    }
}

__global__ void QuadKernel(const scalar_t* a, scalar_t* out, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        out[tid] = a[tid] * a[tid] * a[tid] * a[tid];
    }
}

__global__ void PentaKernel(const scalar_t* a, scalar_t* out, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        out[tid] = a[tid] * a[tid] * a[tid] * a[tid] * a[tid];
    }
}

__global__ void HexaKernel(const scalar_t* a, scalar_t* out, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        out[tid] = a[tid] * a[tid] * a[tid] * a[tid] * a[tid] * a[tid];
    }
}

__global__ void ScalarPowerKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        // see https://docs.nvidia.com/cuda/cuda-math-api/group__CUDA__MATH__INTRINSIC__SINGLE.html#group__CUDA__MATH__INTRINSIC__SINGLE_1g2c2b295816185f6ce2423471df529974
        out[tid] = __powf(a[tid], val);
    }
}

__global__ void ScalarMaximumKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        out[tid] = fmaxf(a[tid], val);
    }
}

__global__ void ScalarEqKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        out[tid] = (a[tid] == val ? 1.0f : 0.0f);
    }
}

__global__ void ScalarGeKernel(const scalar_t* a, scalar_t val, scalar_t* out, size_t size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        out[tid] = (a[tid] >= val ? 1.0f : 0.0f);
    }
}


void EwiseMul    (const CudaArray& a, const CudaArray& b, CudaArray* out) { EWISE_BIN_OP(a, b, out, EwiseMulKernel);  }
void EwiseDiv    (const CudaArray& a, const CudaArray& b, CudaArray* out) { EWISE_BIN_OP(a, b, out, EwiseDivKernel);  }
void EwiseMaximum(const CudaArray& a, const CudaArray& b, CudaArray* out) { EWISE_BIN_OP(a, b, out, EwiseMaximumKernel); }
void EwiseEq     (const CudaArray& a, const CudaArray& b, CudaArray* out) { EWISE_BIN_OP(a, b, out, EwiseEqKernel);   }
void EwiseGe     (const CudaArray& a, const CudaArray& b, CudaArray* out) { EWISE_BIN_OP(a, b, out, EwiseGeKernel);   }
void EwiseLog    (const CudaArray& a,                     CudaArray* out) { EWISE_OP    (a,    out, EwiseLogKernel);  }
void EwiseExp    (const CudaArray& a,                     CudaArray* out) { EWISE_OP    (a,    out, EwiseExpKernel);  }
void EwiseTanh   (const CudaArray& a,                     CudaArray* out) { EWISE_OP    (a,    out, EwiseTanhKernel); }

void ScalarMul    (const CudaArray& a, scalar_t val, CudaArray* out) { SCALAR_OP(a, val, out, ScalarMulKernel); }
void ScalarDiv    (const CudaArray& a, scalar_t val, CudaArray* out) { SCALAR_OP(a, val, out, ScalarDivKernel); }
void ScalarMaximum(const CudaArray& a, scalar_t val, CudaArray* out) { SCALAR_OP(a, val, out, ScalarMaximumKernel); }
void ScalarEq     (const CudaArray& a, scalar_t val, CudaArray* out) { SCALAR_OP(a, val, out, ScalarEqKernel);  }
void ScalarGe     (const CudaArray& a, scalar_t val, CudaArray* out) { SCALAR_OP(a, val, out, ScalarGeKernel);  }
void ScalarPower  (const CudaArray& a, scalar_t val, CudaArray* out)
{
    CudaDims dims = CudaOneDim(out->size);
    if      (std::abs(val + 1.0f     ) < EPS) { InvKernel     <<<dims.grid, dims.block>>>(a.ptr, out->ptr, out->size); }
    else if (std::abs(val + 0.5f     ) < EPS) { InvSqrtKernel <<<dims.grid, dims.block>>>(a.ptr, out->ptr, out->size); }
    else if (std::abs(val + 1.0f/3.0f) < EPS) { InvCbrtKernel <<<dims.grid, dims.block>>>(a.ptr, out->ptr, out->size); }
    else if (std::abs(val)             < EPS) { IdentityKernel<<<dims.grid, dims.block>>>(a.ptr, out->ptr, out->size); }
    else if (std::abs(val - 1.0f/3.0f) < EPS) { CbrtKernel    <<<dims.grid, dims.block>>>(a.ptr, out->ptr, out->size); }
    else if (std::abs(val - 0.5f     ) < EPS) { SqrtKernel    <<<dims.grid, dims.block>>>(a.ptr, out->ptr, out->size); }
    else if (std::abs(val - 1.0f     ) < EPS) { CopyKernel    <<<dims.grid, dims.block>>>(a.ptr, out->ptr, out->size); }
    else if (std::abs(val - 2.0f     ) < EPS) { SquareKernel  <<<dims.grid, dims.block>>>(a.ptr, out->ptr, out->size); }
    else if (std::abs(val - 3.0f     ) < EPS) { CubeKernel    <<<dims.grid, dims.block>>>(a.ptr, out->ptr, out->size); }
    else if (std::abs(val - 4.0f     ) < EPS) { QuadKernel    <<<dims.grid, dims.block>>>(a.ptr, out->ptr, out->size); }
    else if (std::abs(val - 5.0f     ) < EPS) { PentaKernel   <<<dims.grid, dims.block>>>(a.ptr, out->ptr, out->size); }
    else if (std::abs(val - 6.0f     ) < EPS) { HexaKernel    <<<dims.grid, dims.block>>>(a.ptr, out->ptr, out->size); }
    else                                    { ScalarPowerKernel<<<dims.grid, dims.block>>>(a.ptr, val, out->ptr, out->size); }
}

/// END YOUR SOLUTION

////////////////////////////////////////////////////////////////////////////////
// Elementwise and scalar operations
////////////////////////////////////////////////////////////////////////////////

__global__ void MatMulNaiveKernel(const scalar_t* a, const scalar_t* b, scalar_t* out,
                                  uint32_t M, uint32_t N, uint32_t P)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < M * P) {
        size_t x = tid % P,
               y = tid / P;
        scalar_t sum = 0.0f;
        for (uint32_t i = 0; i < N; i++) {
            sum += a[y * N + i] * b[i * P + x];
        }
        out[tid] = sum;
    }
}

// Each block contains `BASE_THREAD_NUM` (256) threads,
// which correponds to 256 elements in a 16 x 16 tile.
#define BLOCK 16

__global__ void MatmulRegTileKernel(const scalar_t* a, const scalar_t* b, scalar_t* out,
                                    uint32_t M, uint32_t N, uint32_t P)
{
    // TODO
}

// see 2nd example in https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
__global__ void MatmulSharedMemTileV0Kernel(const scalar_t* a, const scalar_t* b, scalar_t* out,
                                            uint32_t M, uint32_t N, uint32_t P)
{
    // determine block position
    uint32_t xBlock = blockIdx.x, yBlock = blockIdx.y,
    // determine relative position
             xTile = threadIdx.x, yTile = threadIdx.y,
    // determine absolute position
             x = xBlock * BLOCK + xTile,
             y = yBlock * BLOCK + yTile;
    // compute submatrix
    scalar_t entry = 0.0f;
    for (uint32_t i = 0; i < (N + BLOCK-1) / BLOCK; i++)
    {
        // determine submatrix position
        const scalar_t *subA = &a[yBlock * BLOCK * N + i      * BLOCK],
                       *subB = &b[i      * BLOCK * P + xBlock * BLOCK];
        // load submatrices to shared memory
        __shared__ scalar_t aTile[BLOCK][BLOCK],
                            bTile[BLOCK][BLOCK];
        aTile[yTile][xTile] = subA[yTile * N + xTile];
        bTile[yTile][xTile] = subB[yTile * P + xTile];
        // make sure submatrices are loaded before computation
        __syncthreads();

        for (uint32_t j = 0; j < BLOCK; j++) {
            entry += aTile[yTile][j] * bTile[j][xTile]; // TODO: corner cases
        }
        // make sure computation completes before loading
        __syncthreads();
    }
    //printf("%d, %d: %f\n", x, y, entry);
    out[y * P + x] = entry;
}

// see https://zhuanlan.zhihu.com/p/518857175
__global__ void MatmulSharedMemTileV1Kernel(const scalar_t* a, const scalar_t* b, scalar_t* out,
                                            uint32_t M, uint32_t N, uint32_t P)
{
    // TODO
}

void Matmul(const CudaArray& a, const CudaArray& b, CudaArray* out,
            uint32_t M, uint32_t N, uint32_t P)
{
    /**
     * Multiply two (compact) matrices into an output (also comapct) matrix. You
     * will want to look at the lecture and notes on GPU-based linear algebra to
     * see how to do this. Since ultimately mugrade is just evaluating
     * correctness, you _can_ implement a version that simply parallelizes over
     * (i,j) entries in the output array. However, to really get the full benefit
     * of this problem, we would encourage you to use cooperative fetching,
     * shared memory register tiling, and other ideas covered in the class notes.
     * Note that unlike the tiled matmul function in the CPU backend, here you
     * should implement a single function that works across all size matrices,
     * whether or not they are a multiple of a tile size. As with previous CUDA
     * implementations, this function here will largely just set up the kernel
     * call, and you should implement the logic in a separate MatmulKernel() call.
     * 
     *
     * Args:
     *   a:   compact 2D array of size m x n
     *   b:   comapct 2D array of size n x p
     *   out: compact 2D array of size m x p to write the output to
     *   M: rows    of a / out
     *   N: columns of a / rows of b
     *   P: columns of b / out
     */

    /// BEGIN YOUR SOLUTION
    /// naive implementation
    /*
    CudaDims dims = CudaOneDim(M * P);
    MatMulNaiveKernel<<<dims.grid, dims.block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
    */
    
    /// thread level: register tiling
    // TODO

    /// block level: shared memory tiling
    // WARNING
    // Beware of corner cases where matrix dimensions are not multiples of
    // `BLOCK`. This affects indices of elements.
    // v0
    dim3 grid((P + BLOCK-1) / BLOCK, (M + BLOCK-1) / BLOCK),
         block(BLOCK, BLOCK);
    MatmulSharedMemTileV0Kernel<<<grid, block>>>(a.ptr, b.ptr, out->ptr, M, N, P);
    // v1
    // Each block computes a 128 x 128 tile, and
    // each thread computes an 8 x 8 patch in the tile.
    // Thus, there are 256 threads per block.

    /// END YOUR SOLUTION
}

////////////////////////////////////////////////////////////////////////////////
// Max and sum reductions
////////////////////////////////////////////////////////////////////////////////

__global__ void MaxReduceKernel(const scalar_t* a, scalar_t* out,
                                size_t size, size_t reduce_size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size)
    {
        scalar_t max = a[tid * reduce_size];
        for (size_t i = 1; i < reduce_size; i++) {
            max = max > a[tid * reduce_size + i] ? max : a[tid * reduce_size + i];
        }
        out[tid] = max;
    }
}

void ReduceMax(const CudaArray& a, CudaArray* out, size_t reduce_size)
{
    /**
     * Reduce by taking maximum over `reduce_size` contiguous blocks.  Even though it is inefficient,
     * for simplicity you can perform each reduction in a single CUDA thread.
     * 
     * Args:
     *   a: compact array of size a.size = out.size * reduce_size to reduce over
     *   out: compact array to write into
     *   redice_size: size of the dimension to reduce over
     */
    /// BEGIN YOUR SOLUTION
    CudaDims dim = CudaOneDim(out->size);
    MaxReduceKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr,
                                             a.size, reduce_size);
    /// END YOUR SOLUTION
}


__global__ void ReduceSumKernel(const scalar_t* a, scalar_t* out,
                                size_t size, size_t reduce_size)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < size) {
        scalar_t sum = 0.0f;
        for (size_t i = 0; i < reduce_size; i++) {
            sum += a[tid * reduce_size + i];
        }
        out[tid] = sum;
    }
}

void ReduceSum(const CudaArray& a, CudaArray* out, size_t reduce_size)
{
    /**
     * Reduce by taking summation over `reduce_size` contiguous blocks. Again,
     * for simplicity you can perform each reduction in a single CUDA thread.
     * 
     * Args:
     *   a: compact array of size a.size = out.size * reduce_size to reduce over
     *   out: compact array to write into
     *   redice_size: size of the dimension to reduce over
     */
    /// BEGIN YOUR SOLUTION
    CudaDims dim = CudaOneDim(out->size);
    ReduceSumKernel<<<dim.grid, dim.block>>>(a.ptr, out->ptr, 
                                             out->size, reduce_size);
    /// END YOUR SOLUTION
}


}  // namespace cuda
}  // namespace needle


PYBIND11_MODULE(ndarray_backend_cuda, m)
{
  namespace py = pybind11;
  using namespace needle;
  using namespace cuda;

  m.attr("__device_name__") = "cuda";
  m.attr("__tile_size__") = TILE;

  py::class_<CudaArray>(m, "Array")
      .def(py::init<size_t>(), py::return_value_policy::take_ownership)
      .def_readonly("size", &CudaArray::size)
      .def("ptr", &CudaArray::ptr_as_int);

  // return numpy array, copying from CPU
  m.def("to_numpy", [](const CudaArray& a, std::vector<size_t> shape, std::vector<size_t> strides,
                       size_t offset) {
    std::vector<size_t> numpy_strides = strides;
    std::transform(numpy_strides.begin(), numpy_strides.end(), numpy_strides.begin(),
                   [](size_t& c) { return c * ELEM_SIZE; });

    // copy memory to host
    scalar_t* host_ptr = (scalar_t*)std::malloc(a.size * ELEM_SIZE);
    if (host_ptr == 0) throw std::bad_alloc();
    cudaError_t err = cudaMemcpy(host_ptr, a.ptr, a.size * ELEM_SIZE, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));

    // return numpy array
    py::capsule deallocate_buffer(host_ptr, [](void* p) { free(p); });
    return py::array_t<scalar_t>(shape, numpy_strides, host_ptr + offset, deallocate_buffer);
  });

  // copy numpy array to GPU
  m.def("from_numpy", [](py::array_t<scalar_t> a, CudaArray* out) {
    cudaError_t err =
        cudaMemcpy(out->ptr, a.request().ptr, out->size * ELEM_SIZE, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) throw std::runtime_error(cudaGetErrorString(err));
  });

    m.def("fill", Fill);
    m.def("compact", Compact);
    m.def("ewise_setitem" , EwiseSetitem );
    m.def("scalar_setitem", ScalarSetitem);

    m.def("ewise_add" , EwiseAdd );
    m.def("scalar_add", ScalarAdd);
    m.def("ewise_mul" , EwiseMul );
    m.def("scalar_mul", ScalarMul);
    m.def("ewise_div" , EwiseDiv );
    m.def("scalar_div", ScalarDiv);
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

    m.def("reduce_max", ReduceMax);
    m.def("reduce_sum", ReduceSum);
}
