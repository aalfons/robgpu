
#include "reduction_tools.h"
#include "tool_kernels.h"
#include "colwisemean.h"
#include "colwisesum_kernel.h"

// usage: internal function (expects inputdata on device)
// parameter:
// x: double pointer to matrix (on the device)
// nCols: number of columns in matrix
// nRows: number of rows in matrix
// means: vector for the output (size= nCols * sizeof(double)) on device
__host__ void colwisemean_internal(double * d_x, size_t nCols, size_t nRows, double * d_means)
{
  // predefined settings
  int maxBlocks = 64;
  int maxThreads = 256;

  // to be computed 
  int numBlocks = 0;
  int numThreads = 0;

  getNumBlocksAndThreads(nRows, maxBlocks, maxThreads, numBlocks, numThreads);

  double* d_odata = NULL;
  cudaMalloc((void**) &d_odata, numBlocks*nCols*sizeof(double));

  //execute the kernel
  colwisesum_wrapper<double>(nRows, nCols, numThreads, numBlocks, d_x, d_odata);	

  // sum partial block sums on GPU
  int s=numBlocks;
  int cpuFinalThreshold = 1;
      while(s > cpuFinalThreshold) 
  {
    int threads = 0, blocks = 0;
    getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks, threads);

    colwisesum_wrapper<double>(s, nCols, threads, blocks, d_odata, d_odata);
              
    s = (s + (threads*2-1)) / (threads*2);
  }
          
  // copy final sum from device to device 
  cudaMemcpy( d_means, d_odata, nCols*sizeof(double), cudaMemcpyDeviceToDevice);

  // TODO: optimize!!!
  divide_by_value_kernel<<< nCols/128 + 1, 128>>>(d_means, nCols, nRows);

  cudaFree(d_odata);
}

// usage: 'stand-alone' function (allocates inputdata on device)
// parameter:
// x: double pointer to the R matrix object
// nCols: number of columns in matrix
// nRows: number of rows in matrix
// means: vector for the output (size= nCols * sizeof(double))
__host__ void colwisemean(const double * x, size_t nCols, size_t nRows, double * means)
{
  // predefined settings
  int maxBlocks = 64;
  int maxThreads = 256;

  // to be computed 
  int numBlocks = 0;
  int numThreads = 0;

  getNumBlocksAndThreads(nRows, maxBlocks, maxThreads, numBlocks, numThreads);

  double* d_idata = NULL;
  double* d_odata = NULL;
  cudaMalloc((void**) &d_idata, nCols*nRows*sizeof(double));
  cudaMalloc((void**) &d_odata, numBlocks*nCols*sizeof(double));

  // copy data directly to device memory
  cudaMemcpy(d_idata, x, nCols*nRows*sizeof(double), cudaMemcpyHostToDevice);

  //execute the kernel
  colwisesum_wrapper<double>(nRows, nCols, numThreads, numBlocks, d_idata, d_odata);	

    // sum partial block sums on GPU
  int s=numBlocks;
  int cpuFinalThreshold = 1;
  while(s > cpuFinalThreshold) 
  {
    int threads = 0, blocks = 0;
    getNumBlocksAndThreads(s, maxBlocks, maxThreads, blocks, threads);

    colwisesum_wrapper<double>(s, nCols, threads, blocks, d_odata, d_odata);
              
    s = (s + (threads*2-1)) / (threads*2);
  }

  // divide by nRows TODO: optimize
  divide_by_value_kernel<<< nCols/128 + 1, 128>>>(d_odata, nCols, nRows);

  // copy final sum from device to host
  cudaMemcpy( means, d_odata, nCols*sizeof(double), cudaMemcpyDeviceToHost);

  cudaFree(d_idata);
  cudaFree(d_odata);
}

