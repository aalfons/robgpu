
#include "reduction_tools.h"
#include "tool_kernels.h"
#include "colwisesd_kernel.h"
#include "colwisesum_kernel.h"
#include "colwisemean.h"
#include "colwisesd.h"

// usage: internal function (expects inputdata on device)
// parameter:
// d_x: double pointer to matrix (on the device)
// nCols: number of columns in matrix
// nRows: number of rows in matrix
// output:
// d_means: vector of means for the output (size= nCols * sizeof(double))
// d_sds: vector for the output (size= nCols * sizeof(double)) on device
__host__ void colwisesd_internal(double * d_x, size_t nCols, size_t nRows, double * d_sds)
{

  double * d_means = NULL;
  cudaMalloc((void**) &d_means, nCols * sizeof(double));

  // compute colwise means
  colwisemean_internal( d_x, nCols, nRows, d_means);

  int maxBlocks = 1;
  int maxThreads = 256;

  // to be computed 
  int numBlocks = 0;
  int numThreads = 0;

  getNumBlocksAndThreads(nRows, maxBlocks, maxThreads, numBlocks, numThreads);

  dim3 blocks(numBlocks, nCols, 1);

	// center the input data columnwise 
	columnwisecenter_kernel<<< blocks, numThreads>>>(d_x, nRows, d_means);
	cudaFree(d_means);

	maxBlocks = 64;
	maxThreads = 256;

	// to be computed 
	numBlocks = 0;
	numThreads = 0;

	getNumBlocksAndThreads(nRows, maxBlocks, maxThreads, numBlocks, numThreads);

	// device output data
	double* d_odata = NULL;
	cudaMalloc((void**) &d_odata, numBlocks*nCols*sizeof(double));

	// start the SD computation
	// execute the sd kernel - called via wrapper function
	colwisesd_wrapper<double>(nRows, nCols, numThreads, numBlocks, d_x, d_odata);	
			
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
            
	// sqrt and TODO: optimize
	sqrt_kernel<<< nCols/128 + 1, 128>>>(d_odata, nCols);

    // copy final sum from device to host
  cudaMemcpy( d_sds, d_odata, nCols*sizeof(double), cudaMemcpyDeviceToDevice);
		
	cudaFree(d_odata);
}

// usage: 'stand-alone' function (allocates inputdata on device)
// parameter:
// x: double pointer to the R matrix object
// nCols: number of columns in matrix
// nRows: number of rows in matrix
// sds: vector for the output (size= nCols * sizeof(double))
__host__ void colwisesd(const double * x, size_t nCols, size_t nRows, double * sds)
{

  double* d_idata = NULL;
  double* d_means = NULL;
	double* d_sds = NULL;
	
	cudaMalloc((void**) &d_idata, nCols*nRows*sizeof(double));
  cudaMalloc((void**) &d_means, nCols*sizeof(double));
  cudaMalloc((void**) &d_sds, nCols*sizeof(double));

	// copy input data to device memory
  cudaMemcpy(d_idata, x, nCols*nRows*sizeof(double), cudaMemcpyHostToDevice);

	// compute colwise means
	colwisemean_internal( d_idata, nCols, nRows, d_means);

  int maxBlocks = 1;
  int maxThreads = 256;

  // to be computed 
  int numBlocks = 0;
  int numThreads = 0;

  getNumBlocksAndThreads(nRows, maxBlocks, maxThreads, numBlocks, numThreads);

  dim3 blocks(numBlocks, nCols, 1);

	// center the input data columnwise 
	columnwisecenter_kernel<<< blocks, numThreads>>>(d_idata, nRows, d_means);
	cudaFree(d_means);

	maxBlocks = 64;
	maxThreads = 256;

	// to be computed 
	numBlocks = 0;
	numThreads = 0;

	getNumBlocksAndThreads(nRows, maxBlocks, maxThreads, numBlocks, numThreads);

	double* d_odata = NULL;
	cudaMalloc((void**) &d_odata, numBlocks*nCols*sizeof(double));

	// start the SD computation
	// execute the sd kernel - called via wrapper function
	colwisesd_wrapper<double>(nRows, nCols, numThreads, numBlocks, d_idata, d_odata);	
			
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
            
	// sqrt and divide by nRows-1 TODO: optimize
	sqrt_divide_by_value_kernel<<< nCols/128 + 1, 128>>>(d_odata, nCols, nRows - 1);

  // copy final sum from device to host
  cudaMemcpy( sds, d_odata, nCols*sizeof(double), cudaMemcpyDeviceToHost);
		

	cudaFree(d_idata);
	cudaFree(d_odata);
	cudaFree(d_sds);

}

