#include "robgpu_settings.h"

#include <R.h>
#include "reduction.h"
#include "colwisemean.h"
#include "colwisesd.h"
#include "tools.h"

// TODO: OPTIMIZE!!! sqrt_divide-by-nRow kernel
__global__ void sqrt_kernel(MYTYPE * d_x, size_t n)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	if(i < n)
		d_x[i] = sqrt(d_x[i]);
}

// TODO: OPTIMIZE!!! sqrt_divide-by-nRow kernel
__global__ void sqrt_divide_by_value_kernel(MYTYPE * d_x, size_t n, int value)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	if(i < n)
		d_x[i] = sqrt(d_x[i] / (MYTYPE) value);
}


// usage: internal function (expects inputdata on device)
// parameter:
// d_x: MYTYPE pointer to matrix (on the device)
// nCols: number of columns in matrix
// nRows: number of rows in matrix
// output:
// d_means: vector of means for the output (size= nCols * sizeof(MYTYPE))
// d_sds: vector for the output (size= nCols * sizeof(MYTYPE)) on device
__host__ void colwisesd_internal(MYTYPE * d_x, size_t nCols, size_t nRows, MYTYPE * d_sds)
{

  MYTYPE * d_means = NULL;
  cudaMalloc((void**) &d_means, nCols * sizeof(MYTYPE));

  // compute colwise means
  colwisemean_internal( d_x, nCols, nRows, d_means);

  int maxBlocks = 1;
  int maxThreads = 256;
  int kernel = 7; // 7 is colwisemean

  // to be computed 
  int numBlocks = 0;
  int numThreads = 0;

  getNumBlocksAndThreads(kernel, nRows, maxBlocks, maxThreads, numBlocks, numThreads);

  dim3 blocks(numBlocks, nCols, 1);

	// center the input data columnwise 
	columnwisecenter_kernel<<< blocks, numThreads>>>(d_x, nRows, d_means);
	cudaFree(d_means);

	maxBlocks = 64;
	maxThreads = 256;
	kernel = 7; // 7 is colwisemean

	// to be computed 
	numBlocks = 0;
	numThreads = 0;

	getNumBlocksAndThreads(kernel, nRows, maxBlocks, maxThreads, numBlocks, numThreads);

	// device output data
	MYTYPE* d_odata = NULL;
	cudaMalloc((void**) &d_odata, numBlocks*nCols*sizeof(MYTYPE));

	// start the SD computation
	// execute the sd kernel - called via wrapper function
	colwisesd_wrapper<MYTYPE>(nRows, nCols, numThreads, numBlocks, d_x, d_odata);	
			
	// sum partial block sums on GPU
	int s=numBlocks;
	int cpuFinalThreshold = 1;
    while(s > cpuFinalThreshold) 
	{
		int threads = 0, blocks = 0;
		getNumBlocksAndThreads(kernel, s, maxBlocks, maxThreads, blocks, threads);

		//reduce<float>(s, threads, blocks, 6, d_odata, d_odata);
		colwisesum_wrapper<MYTYPE>(s, nCols, threads, blocks, d_odata, d_odata);
               
		s = (s + (threads*2-1)) / (threads*2);
	}
            
	cudaThreadSynchronize();

	// sqrt and TODO: optimize
	sqrt_kernel<<< nCols/128 + 1, 128>>>(d_odata, nCols);

    // copy final sum from device to host
  cudaMemcpy( d_sds, d_odata, nCols*sizeof(MYTYPE), cudaMemcpyDeviceToDevice);
		
	cudaFree(d_odata);
}

// usage: 'stand-alone' function (allocates inputdata on device)
// parameter:
// x: MYTYPE pointer to the R matrix object
// nCols: number of columns in matrix
// nRows: number of rows in matrix
// sds: vector for the output (size= nCols * sizeof(MYTYPE))
__host__ void colwisesd(const MYTYPE * x, size_t nCols, size_t nRows, MYTYPE * sds)
{
	//printf("\n sorry, tut noch nix... :-) \n");

    MYTYPE* d_idata = NULL;
    MYTYPE* d_means = NULL;
	MYTYPE* d_sds = NULL;
	
	cudaMalloc((void**) &d_idata, nCols*nRows*sizeof(MYTYPE));
    cudaMalloc((void**) &d_means, nCols*sizeof(MYTYPE));
    cudaMalloc((void**) &d_sds, nCols*sizeof(MYTYPE));

	// copy input data to device memory
    cudaMemcpy(d_idata, x, nCols*nRows*sizeof(MYTYPE), cudaMemcpyHostToDevice);

	// compute colwise means
	colwisemean_internal( d_idata, nCols, nRows, d_means);

  int maxBlocks = 1;
  int maxThreads = 256;
  int kernel = 7; // 7 is colwisemean

  // to be computed 
  int numBlocks = 0;
  int numThreads = 0;

  getNumBlocksAndThreads(kernel, nRows, maxBlocks, maxThreads, numBlocks, numThreads);

  dim3 blocks(numBlocks, nCols, 1);

	// center the input data columnwise 
	columnwisecenter_kernel<<< blocks, numThreads>>>(d_idata, nRows, d_means);
	cudaFree(d_means);

	maxBlocks = 64;
	maxThreads = 256;
	kernel = 7; // 7 is colwisemean

	// to be computed 
	numBlocks = 0;
	numThreads = 0;

	getNumBlocksAndThreads(kernel, nRows, maxBlocks, maxThreads, numBlocks, numThreads);

	MYTYPE* d_odata = NULL;
	cudaMalloc((void**) &d_odata, numBlocks*nCols*sizeof(MYTYPE));

	// start the SD computation
	// execute the sd kernel - called via wrapper function
	colwisesd_wrapper<MYTYPE>(nRows, nCols, numThreads, numBlocks, d_idata, d_odata);	
			
	// sum partial block sums on GPU
	int s=numBlocks;
	int cpuFinalThreshold = 1;
    while(s > cpuFinalThreshold) 
	{
		int threads = 0, blocks = 0;
		getNumBlocksAndThreads(kernel, s, maxBlocks, maxThreads, blocks, threads);

		//reduce<MYTYPE>(s, threads, blocks, 6, d_odata, d_odata);
		colwisesum_wrapper<MYTYPE>(s, nCols, threads, blocks, d_odata, d_odata);
               
		s = (s + (threads*2-1)) / (threads*2);
	}
            
	cudaThreadSynchronize();

	// sqrt and divide by nRows-1 TODO: optimize
	sqrt_divide_by_value_kernel<<< nCols/128 + 1, 128>>>(d_odata, nCols, nRows - 1);

    // copy final sum from device to host
    cudaMemcpy( sds, d_odata, nCols*sizeof(MYTYPE), cudaMemcpyDeviceToHost);
		

	cudaFree(d_idata);
	cudaFree(d_odata);
	cudaFree(d_sds);

}

