
#ifndef _robgpu_TOOL_KERNELS_H_
#define _robgpu_TOOL_KERNELS_H_

// kernel to be optimized to divide each element by value
__global__ void divide_by_value_kernel(double * d_x, size_t n, size_t value);

// TODO: OPTIMIZE!!! sqrt_divide-by-nRow kernel
__global__ void sqrt_kernel(double * d_x, size_t n);

// TODO: OPTIMIZE!!! sqrt_divide-by-nRow kernel
__global__ void sqrt_divide_by_value_kernel(double * d_x, size_t n, int value);

// TODO: OPTIMIZE!!!
__global__ void columnwisecenter_kernel(double * d_x, size_t n, double * d_col_means);

// TODO: OPTIMIZE!!!
__global__ void subsample_kernel(double * d_x, size_t n, double * d_x_subsample, size_t sample_size, int * d_sample_index, int nsamp, int actual_sample);

// TODO: OPTIMIZE!!!
// divide correlation matrix elements by sd(x)*sd(y)
__global__ void divide_by_value_indexed_kernel(double * d_cors, double * d_x_sds, double * d_y_sds);

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T*()
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T*() const
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};

// specialize for double to avoid unaligned memory 
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double*()
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }

    __device__ inline operator const double*() const
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }
};

#endif // _robgpu_TOOL_KERNELS_H_ 

