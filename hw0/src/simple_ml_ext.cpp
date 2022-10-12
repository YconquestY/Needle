#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <math.h> // `ceil`, `exp`
#include <iostream>

namespace py = pybind11;


void gemm(float *A, float *B, float *C,
          size_t a, size_t b, size_t c)
{
    /**
     * A C++ version of sequential matrix multiplication.
     * Tiling is not adopted provided the small input size.
     * 
     * Args:
     *     A (float *): pointer to the first matrix of size a x b,
     *                  stored in row-major order
     *     B (float *): pointer to the second matrix of size b x c,
     *                  stored in row-major order
     *     C (float *): pointer to the matrix product of size a x c,
     *                  stored in row-major order
     *     a (size_t) : number of rows in A
     *     b (size_t) : number of columns in A
     *                  number of rows in B
     *     c (size_t) : number of columns in B
     * 
     * Returns:
     *     (None)
     */
    for (unsigned int i = 0; i < a; i++)
    {
        for (unsigned int j = 0; j < c; j++)
        {
            float entry = 0.0f;
            for (unsigned int k = 0; k < b; k++) {
                entry += (float) A[i * b + k] * (float) B[k * c + j];
            }
            C[i * c + j] = entry;
        }
    }
}

void transpose(float *A, float *A_T,
               size_t a, size_t b)
{
    /**
     * A C++ version of matrix transpose.
     * 
     * Args:
     *     A (float *)  : pointer to the matrix of size a x b,
     *                    stored in row-major order
     *     A_T (float *): pointer to the transposed matrix of size b x a,
     *                    stored in row-major order
     *     a (size_t) : number of rows in A
     *     b (size_t) : number of columns in A
     * 
     * Returns:
     *     (None)
     */
    for (unsigned int i = 0; i < a; i++)
    {
        for (unsigned int j = 0; j < b; j++)
        {
            A_T[j * a + i] = A[i * b + j];
        }
    }
}

void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *                        major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (foat *): pointer to theta data, of size n*k, stored in row
     *                     major (C) format
     *     m (size_t): number of exmaples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    int num_iter = (int) ceil((float) m / (float) batch);
    float *Z, *Z_sum, *I, *X_batch_T, *grad; // declare matrices/vectors
    for (int i = 0; i < num_iter; i++)
    {
        /* get batch */
        int batch_size = batch;
        if (i == num_iter - 1) {
            batch_size = (int) m - i * batch;
        }
        // pointers to start of batch
        float         *X_batch = (float *)         &X[i * batch * n];
        unsigned char *y_batch = (unsigned char *) &y[i * batch]; // An (unsigned) `char` occupies 1B while an `int` occupies 4B.
        /* compute gradient */
        Z = (float *) malloc(sizeof(float) * batch_size * k);
        gemm(X_batch, theta, Z, (size_t) batch_size, n, k);
        for (unsigned int i = 0; i < ((size_t) batch_size) * k; i++) {
            Z[i] = exp(Z[i]);
        }
        Z_sum = (float *) malloc(sizeof(float) * batch_size);
        for (unsigned int i = 0; i < (size_t) batch_size; i++)
        {
            float sum = 0.0f;
            for (unsigned int j = 0; j < k; j++) {
                sum += Z[i * k + j];
            }
            Z_sum[i] = sum;
        }
        for (unsigned int i = 0; i < ((size_t) batch_size) * k; i++) {
            Z[i] /= Z_sum[i/k];
        }

        I = new float[batch_size * k]();
        for (unsigned int i = 0; i < (size_t) batch_size; i++) {
            I[i * k + (unsigned int) y_batch[i]] = 1.0;
        }

        for (unsigned int i = 0; i < ((size_t) batch_size) * k; i++) {
            Z[i] -= I[i];
        }
        X_batch_T = (float *) malloc(sizeof(float) * n * batch_size);
        transpose(X_batch, X_batch_T, (size_t) batch_size, n);
        grad = (float *) malloc(sizeof(float) * n * k);
        gemm(X_batch_T, Z, grad, n, (size_t) batch_size, k);
        for (unsigned int i = 0; i < n * k; i++) {
            grad[i] /= (float) batch_size;
        }
        /* gradient descent */
        for (unsigned int i = 0; i < n * k; i++) {
            theta[i] -= lr * grad[i];
        }
        /* free memory */
        free(Z);
        free(Z_sum);
        delete(I);
        free(X_batch_T);
        free(grad);
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
