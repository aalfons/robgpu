
#include "reduction_tools.h"
#include "colwisesum.h"
#include "colwisesum_kernel.h"

// usage: 'stand-alone' function (allocates inputdata on device)
// parameter:
// x: double pointer to the R matrix object
// nCols: number of columns in matrix
// nRows: number of rows in matrix
// sums: vector for final results
__host__ void colwisesum(const double * x, 
                         size_t nCols, 
                         size_t nRows, 
                         double * sums)
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
		
 // copy data to device memory
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
  
  // copy final sum from device to host
  cudaMemcpy( sums, d_odata, nCols*sizeof(double), cudaMemcpyDeviceToHost);
		
  cudaFree(d_idata);
  cudaFree(d_odata);
}

