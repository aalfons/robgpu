/*
 * Code is adapted from NVIDIA examples
 *
 * See NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include "reduction_tools.h"
#include "tool_kernels.h"
#include "colwisesd_kernel.h"


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
  
  if (tid < 32)
  {
      // now that we are using warp-synchronous programming (below)
      // we need to declare our shared memory volatile so that the compiler
      // doesn't reorder stores to it and induce incorrect behavior.
      volatile T* smem = sdata;
      if (blockSize >=  64) { smem[tid] = mySum = mySum + smem[tid + 32];  }
      if (blockSize >=  32) { smem[tid] = mySum = mySum + smem[tid + 16];  }
      if (blockSize >=  16) { smem[tid] = mySum = mySum + smem[tid +  8];  }
      if (blockSize >=   8) { smem[tid] = mySum = mySum + smem[tid +  4];  }
      if (blockSize >=   4) { smem[tid] = mySum = mySum + smem[tid +  2];  }
      if (blockSize >=   2) { smem[tid] = mySum = mySum + smem[tid +  1];  }
  }
  
  if (tid == 0) 
      g_odata[blockIdx.y*gridDim.x + blockIdx.x] = sdata[0]; 
}

////////////////////////////////////////////////////////////////////////////////
// Wrapper function for colwise sd kernel launch
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

// Instantiate the colwisesd_wrapper function for 3 types
template void 
colwisesd_wrapper<int>(int nrows, int ncols, int threads, 
							int blocks, int *d_idata, int *d_odata);

template void 
colwisesd_wrapper<double>(int nrows, int ncols, int threads, 
							int blocks, double *d_idata, double *d_odata);


