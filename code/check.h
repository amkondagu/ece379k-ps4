// check.h -- stop the program on the first CUDA error.
//
// CUDA_CHECK(call)  wraps any runtime call that returns cudaError_t.
// KERNEL_CHECK()    goes after a kernel launch: it reports a bad
//                   launch, then waits for the kernel and reports
//                   any error that happened while it ran.
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                          \
    do {                                                          \
        cudaError_t err_ = (call);                                \
        if (err_ != cudaSuccess) {                                \
            fprintf(stderr, "%s:%d: %s\n  -> %s\n", __FILE__,     \
                    __LINE__, #call, cudaGetErrorString(err_));   \
            exit(1);                                              \
        }                                                         \
    } while (0)

#define KERNEL_CHECK()                                            \
    do {                                                          \
        CUDA_CHECK(cudaGetLastError());                           \
        CUDA_CHECK(cudaDeviceSynchronize());                      \
    } while (0)
