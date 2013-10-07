/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
    Parallel reduction kernels
*/

#ifndef _REDUCE_KERNEL_H_
#define _REDUCE_KERNEL_H_

#include <stdio.h>
#include "reduction.h"

#ifdef __DEVICE_EMULATION__
#define EMUSYNC __syncthreads()
#else
#define EMUSYNC
#endif

#define DEBUG true

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif

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

/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory. 
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.  
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
// colwise sum KERNEL
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
colwisesum_kernel(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
	  unsigned int colid = blockIdx.y;
    unsigned int i = colid*n + blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
   
    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < (colid+1)*n)
    {         
        mySum += (isnan(g_idata[i]) ? 0 : g_idata[i]);
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n * (colid+1)) 
            mySum += (isnan(g_idata[i+blockSize]) ? 0 : g_idata[i+blockSize]);
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }
    
#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T* smem = sdata;
        if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; EMUSYNC; }
        if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; EMUSYNC; }
        if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; EMUSYNC; }
        if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; EMUSYNC; }
        if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; EMUSYNC; }
        if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; EMUSYNC; }
    }
    
    if (tid == 0) 
        g_odata[blockIdx.y*gridDim.x + blockIdx.x] = sdata[0]; //colid*n + blockIdx.x*blockSize*2 + threadIdx.x;//sdata[0];
}

/*
    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)

    Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory. 
    In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.  
    If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
// mahalanobis distance kernel KERNEL
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
mahalanobis_distance_kernel(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
	  unsigned int colid = blockIdx.y;
    unsigned int i = colid*n + blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
   
    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < (colid+1)*n)
    {         
        mySum += (isnan(g_idata[i]) ? 0 : g_idata[i]);
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n * (colid+1)) 
            mySum += (isnan(g_idata[i+blockSize]) ? 0 : g_idata[i+blockSize]);
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    sdata[tid] = mySum;
    __syncthreads();


    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }
    
#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T* smem = sdata;
        if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; EMUSYNC; }
        if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; EMUSYNC; }
        if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; EMUSYNC; }
        if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; EMUSYNC; }
        if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; EMUSYNC; }
        if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; EMUSYNC; }
    }
    
    if (tid == 0) 
        g_odata[blockIdx.y*gridDim.x + blockIdx.x] = sdata[0]; //colid*n + blockIdx.x*blockSize*2 + threadIdx.x;//sdata[0];
}


// SD KERNEL
template <class T, unsigned int blockSize, bool nIsPow2>
__global__ void
colwisesd_kernel(T *g_idata, T *g_odata, unsigned int n)
{
    T *sdata = SharedMemory<T>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
	unsigned int colid = blockIdx.y;
    unsigned int i = colid*n + blockIdx.x*blockSize*2 + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
   
    T mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the 
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < (colid+1)*n)
    {         
        mySum += (g_idata[i] * g_idata[i]);
        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n * (colid+1)) 
            mySum += (g_idata[i+blockSize] * g_idata[i+blockSize]);
        i += gridSize;
    } 

    // each thread puts its local sum into shared memory 
    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512) { if (tid < 256) { sdata[tid] = mySum = mySum + sdata[tid + 256]; } __syncthreads(); }
    if (blockSize >= 256) { if (tid < 128) { sdata[tid] = mySum = mySum + sdata[tid + 128]; } __syncthreads(); }
    if (blockSize >= 128) { if (tid <  64) { sdata[tid] = mySum = mySum + sdata[tid +  64]; } __syncthreads(); }
    
#ifndef __DEVICE_EMULATION__
    if (tid < 32)
#endif
    {
        // now that we are using warp-synchronous programming (below)
        // we need to declare our shared memory volatile so that the compiler
        // doesn't reorder stores to it and induce incorrect behavior.
        volatile T* smem = sdata;
        if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32]; EMUSYNC; }
        if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16]; EMUSYNC; }
        if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8]; EMUSYNC; }
        if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4]; EMUSYNC; }
        if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2]; EMUSYNC; }
        if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1]; EMUSYNC; }
    }
    
    if (tid == 0) 
        g_odata[blockIdx.y*gridDim.x + blockIdx.x] = sdata[0]; 
}

bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

unsigned int nextPow2( unsigned int x ) 
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the given reduction kernel
// For the kernels >= 3, we set threads / block to the minimum of maxThreads and
// n/2. For kernels < 3, we set to the minimum of maxThreads and n.  For kernel 
// 6, we observe the maximum specified number of blocks, because each thread in 
// that kernel can process a variable number of elements.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
    
    
    
    if (whichKernel < 3)
    {
        threads = (n < maxThreads) ? nextPow2(n) : maxThreads;
        blocks = (n + threads - 1) / threads;
    }
    else
    {
        threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
        blocks = (n + (threads * 2 - 1)) / (threads * 2);
    }
        

    if (whichKernel >= 6)
        blocks = MIN(maxBlocks, blocks);
}


////////////////////////////////////////////////////////////////////////////////
// Wrapper function for colwise sum kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void 
colwisesum_wrapper(int nrows, int ncols, int threads, int blocks, 
       T *d_idata, T *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, ncols, 1);
	
    // when there is only one warp per block, we need to allocate two warps 
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    // choose which of the optimized versions of reduction to launch
    if (isPow2(nrows))
    {
        switch (threads)
        {
        case 512:
            colwisesum_kernel<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 256:
            colwisesum_kernel<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break; 
        case 128:
            colwisesum_kernel<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 64:
            colwisesum_kernel<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 32:
            colwisesum_kernel<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 16:
            colwisesum_kernel<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case  8:
            colwisesum_kernel<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case  4:
            colwisesum_kernel<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case  2:
            colwisesum_kernel<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case  1:
            colwisesum_kernel<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        }
    }
    else
    {
        switch (threads)
        {
        case 512:
            colwisesum_kernel<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 256:
            colwisesum_kernel<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 128:
            colwisesum_kernel<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 64:
            colwisesum_kernel<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 32:
            colwisesum_kernel<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 16:
            colwisesum_kernel<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case  8:
            colwisesum_kernel<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case  4:
            colwisesum_kernel<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case  2:
            colwisesum_kernel<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case  1:
            colwisesum_kernel<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        }
    }

}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for mahalanobis distance kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void 
mahalanobis_distance_wrapper(int nrows, int ncols, int threads, int blocks, 
       T *d_idata, T *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
	  dim3 dimGrid(blocks, ncols, 1);
	
    // when there is only one warp per block, we need to allocate two warps 
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    // choose which of the optimized versions of reduction to launch
    if (isPow2(nrows))
    {
        switch (threads)
        {
        case 512:
            mahalanobis_distance_kernel<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 256:
            mahalanobis_distance_kernel<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break; 
        case 128:
            mahalanobis_distance_kernel<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 64:
            mahalanobis_distance_kernel<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 32:
            mahalanobis_distance_kernel<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 16:
            mahalanobis_distance_kernel<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case  8:
            mahalanobis_distance_kernel<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case  4:
            mahalanobis_distance_kernel<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case  2:
            mahalanobis_distance_kernel<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case  1:
            mahalanobis_distance_kernel<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        }
    }
    else
    {
        switch (threads)
        {
        case 512:
            mahalanobis_distance_kernel<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 256:
            mahalanobis_distance_kernel<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 128:
            mahalanobis_distance_kernel<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 64:
            mahalanobis_distance_kernel<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 32:
            mahalanobis_distance_kernel<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 16:
            mahalanobis_distance_kernel<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case  8:
            mahalanobis_distance_kernel<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case  4:
            mahalanobis_distance_kernel<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case  2:
            mahalanobis_distance_kernel<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case  1:
            mahalanobis_distance_kernel<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        }
    }

}



////////////////////////////////////////////////////////////////////////////////
// Wrapper function for colwise mean kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void 
colwisesd_wrapper(int nrows, int ncols, int threads, int blocks, 
       T *d_idata, T *d_odata)
{
    dim3 dimBlock(threads, 1, 1);
	dim3 dimGrid(blocks, ncols, 1);
	
    // when there is only one warp per block, we need to allocate two warps 
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(T) : threads * sizeof(T);

    // choose which of the optimized versions of reduction to launch
    if (isPow2(nrows))
    {
        switch (threads)
        {
        case 512:
            colwisesd_kernel<T, 512, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 256:
            colwisesd_kernel<T, 256, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 128:
            colwisesd_kernel<T, 128, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 64:
            colwisesd_kernel<T,  64, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 32:
            colwisesd_kernel<T,  32, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 16:
            colwisesd_kernel<T,  16, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case  8:
            colwisesd_kernel<T,   8, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case  4:
            colwisesd_kernel<T,   4, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case  2:
            colwisesd_kernel<T,   2, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case  1:
            colwisesd_kernel<T,   1, true><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        }
    }
    else
    {
        switch (threads)
        {
        case 512:
            colwisesd_kernel<T, 512, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 256:
            colwisesd_kernel<T, 256, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 128:
            colwisesd_kernel<T, 128, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 64:
            colwisesd_kernel<T,  64, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 32:
            colwisesd_kernel<T,  32, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case 16:
            colwisesd_kernel<T,  16, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case  8:
            colwisesd_kernel<T,   8, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case  4:
            colwisesd_kernel<T,   4, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case  2:
            colwisesd_kernel<T,   2, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        case  1:
            colwisesd_kernel<T,   1, false><<< dimGrid, dimBlock, smemSize >>>(d_idata, d_odata, nrows); break;
        }
    }
}



// Instantiate the colwisesum_wrapper function for 3 types
template void 
colwisesum_wrapper<int>(int nrows, int ncols, int threads, 
							int blocks, int *d_idata, int *d_odata);

template void 
colwisesum_wrapper<float>(int nrows, int ncols, int threads, 
							int blocks, float *d_idata, float *d_odata);

template void 
colwisesum_wrapper<double>(int nrows, int ncols, int threads, 
							int blocks, double *d_idata, double *d_odata);

// Instantiate the colwisesum_wrapper function for 3 types
template void 
mahalanobis_distance_wrapper<int>(int nrows, int ncols, int threads, 
							int blocks, int *d_idata, int *d_odata);

template void 
mahalanobis_distance_wrapper<float>(int nrows, int ncols, int threads, 
							int blocks, float *d_idata, float *d_odata);

template void 
mahalanobis_distance_wrapper<double>(int nrows, int ncols, int threads, 
							int blocks, double *d_idata, double *d_odata);



// Instantiate the colwisesd_wrapper function for 3 types
template void 
colwisesd_wrapper<int>(int nrows, int ncols, int threads, 
							int blocks, int *d_idata, int *d_odata);

template void 
colwisesd_wrapper<float>(int nrows, int ncols, int threads, 
							int blocks, float *d_idata, float *d_odata);

template void 
colwisesd_wrapper<double>(int nrows, int ncols, int threads, 
							int blocks, double *d_idata, double *d_odata);

#endif // #ifndef _REDUCE_KERNEL_H_
