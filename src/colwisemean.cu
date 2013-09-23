
#include "robgpu_settings.h"

#include<R.h>
#include"reduction.h"
#include"colwisemean.h"

#define DEBUG false

// TODO: OPTIMIZE!!! divide-by-nRow kernel
__global__ void divide_by_value_kernel(MYTYPE * d_x, size_t n, size_t value)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	if(i < n)
		d_x[i] = d_x[i] / ((MYTYPE)value);
}


// usage: internal function (expects inputdata on device)
// parameter:
// x: MYTYPE pointer to matrix (on the device)
// nCols: number of columns in matrix
// nRows: number of rows in matrix
// means: vector for the output (size= nCols * sizeof(MYTYPE)) on device
__host__ void colwisemean_internal(MYTYPE * d_x, size_t nCols, size_t nRows, MYTYPE * d_means)
{
		// predefined settings
		int maxBlocks = 64;
		int maxThreads = 256;
        int kernel = 7; // 7 is colwisemean

		// to be computed 
		int numBlocks = 0;
        int numThreads = 0;

        getNumBlocksAndThreads(kernel, nRows, maxBlocks, maxThreads, numBlocks, numThreads);

		//printf("\ncolwisemean_internal: numBlocks=%d, numThreads=%d\n nCols=%d, nRows=%d", numBlocks, numThreads, nCols, nRows);

        MYTYPE* d_odata = NULL;
        cudaMalloc((void**) &d_odata, numBlocks*nCols*sizeof(MYTYPE));

		//execute the kernel
		//reduce<MYTYPE>(nRows, numThreads, numBlocks, 6, d_idata, d_odata);	
		colwisesum_wrapper<MYTYPE>(nRows, nCols, numThreads, numBlocks, d_x, d_odata);	

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
	
		// copy final sum from device to device 
        cudaMemcpy( d_means, d_odata, nCols*sizeof(MYTYPE), cudaMemcpyDeviceToDevice);

		// TODO: optimize!!!
		divide_by_value_kernel<<< nCols/128 + 1, 128>>>(d_means, nCols, nRows);

		cudaFree(d_odata);
}

// usage: 'stand-alone' function (allocates inputdata on device)
// parameter:
// x: MYTYPE pointer to the R matrix object
// nCols: number of columns in matrix
// nRows: number of rows in matrix
// means: vector for the output (size= nCols * sizeof(MYTYPE))
__host__ void colwisemean(const MYTYPE * x, size_t nCols, size_t nRows, MYTYPE * means)
{
		// predefined settings
		int maxBlocks = 64;
		int maxThreads = 256;
        int kernel = 7; // 7 is colwisemean

		// to be computed 
		int numBlocks = 0;
        int numThreads = 0;

        getNumBlocksAndThreads(kernel, nRows, maxBlocks, maxThreads, numBlocks, numThreads);

        MYTYPE* d_idata = NULL;
        MYTYPE* d_odata = NULL;
		cudaMalloc((void**) &d_idata, nCols*nRows*sizeof(MYTYPE));
        cudaMalloc((void**) &d_odata, numBlocks*nCols*sizeof(MYTYPE));
		
        // copy data directly to device memory
        cudaMemcpy(d_idata, x, nCols*nRows*sizeof(MYTYPE), cudaMemcpyHostToDevice);
		// warum das ist fraglich... ?!? 
        cudaMemcpy(d_odata, x, numBlocks*nCols*sizeof(MYTYPE), cudaMemcpyHostToDevice);

		//execute the kernel
		//reduce<MYTYPE>(nRows, numThreads, numBlocks, 6, d_idata, d_odata);	
		colwisesum_wrapper<MYTYPE>(nRows, nCols, numThreads, numBlocks, d_idata, d_odata);	

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

		// divide by nRows TODO: optimize
		divide_by_value_kernel<<< nCols/128 + 1, 128>>>(d_odata, nCols, nRows);

        // copy final sum from device to host
        cudaMemcpy( means, d_odata, nCols*sizeof(MYTYPE), cudaMemcpyDeviceToHost);
		
		//colwisemean_internal(d_idata, nCols, nRows, d_odata);
        //cudaMemcpy( means, d_odata, nCols*sizeof(MYTYPE), cudaMemcpyDeviceToHost);

		cudaFree(d_idata);
		cudaFree(d_odata);
}

