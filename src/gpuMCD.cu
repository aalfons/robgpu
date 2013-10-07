
#include <stdio.h>

#include "robgpu_settings.h"

#include <R.h>
#include <Rdefines.h>

#include "gpuMCD.h" // needs MYTYPE 
#include "compute_det.h" // needs MYTYPE 
#include "compute_inverse.h" // needs MYTYPE 
#include "mahalanobis_distance.h"

#include "colwisesd.h"
#include "colwisemean.h"
#include "cuseful.h"
#include "tools.h"
#include "reduction.h"

#include <cublas.h>


// divide correlation matrix elements by sd(x)*sd(y)
__global__ void divide_by_value_indexed_kernel(MYTYPE * d_cors, int dim)
{
	d_cors[blockIdx.y*gridDim.x + blockIdx.x] /= ((MYTYPE)(dim - 1));
  // d_cors[blockIdx.y*gridDim.x + blockIdx.x] /= (d_x_sds[blockIdx.x] * d_y_sds[blockIdx.y]);	
}

__host__ void gpuMCD(MYTYPE * h_x, 
                     int n, 
                     int p, 
                     MYTYPE * h_covMat, 
                     SEXP p_covMat, 
                     MYTYPE * h_covMat_det,
                     int nsamp, 
                     int sample_size, 
                     int * p_sample_index, 
                     unsigned int gpuID)
{

  // GPU device memory
	MYTYPE * d_x = NULL;
	MYTYPE * d_x_means = NULL;
	MYTYPE * d_x_subsample = NULL;
  int * d_sample_index = NULL;
	MYTYPE * d_cov = NULL;
  MYTYPE * d_mh_dist = NULL;

  // host memory
  MYTYPE * h_covMat_inverse = NULL;
	
  h_covMat_inverse = (MYTYPE *) calloc (p * p, sizeof(MYTYPE));

  // printf("n: %d p: %d nsamp: %d sample_size: %d\n", n, p, nsamp, sample_size);

  cudaSetDevice(gpuID);

	cublasInit();
   
	checkCublasError("mcd cublas init...");

	
	// allocate device memory
	cublasAlloc(p * n, sizeof(MYTYPE), (void**) &d_x);
  cublasAlloc(p, sizeof(MYTYPE), (void**) &d_x_means);
	cublasAlloc(p * sample_size, sizeof(MYTYPE), (void**) &d_x_subsample);
	cublasAlloc(nsamp * sample_size, sizeof(int), (void**) &d_sample_index);
  cublasAlloc(p*p, sizeof(MYTYPE), (void**) &d_cov);

  cublasAlloc(p, sizeof(MYTYPE), (void**) &d_mh_dist);

	checkCublasError("mcd gpu memory allocation");

	// copy input data to gpu
	cublasSetMatrix(n, p, sizeof(MYTYPE), h_x, n, d_x, n);
	cublasSetMatrix(nsamp, sample_size, sizeof(int), p_sample_index, nsamp, d_sample_index, nsamp);

	checkCublasError("mcd set matrix");

  for (unsigned int i = 0; i < nsamp; i ++) {
	
    // printf("%3d ", i);
    // performance question:
    // better to copy GPU -> GPU or HOST -> GPU
    // problem: limited GPU RAM
    // no subsample-wise -- colwise mean/center etc necessary ->
    //     when using dgemm JUST on the subsample... => GPU -> GPU
    int maxBlocks = 1;
    int maxThreads = 256;
    int kernel = 7; // 7 is colwisemean

    // to be computed 
    int numBlocks = 0;
    int numThreads = 0;

    // TODO resolve strange getNumBlocks... etc computation
    getNumBlocksAndThreads(kernel, sample_size, maxBlocks, maxThreads, numBlocks, numThreads);

    dim3 blocks(numBlocks, p, 1);

    // printf("numBlocks: %d, numThreads: %d \n", numBlocks, numThreads);
    subsample_kernel<<< blocks, numThreads>>>(d_x, n, d_x_subsample, sample_size, d_sample_index, nsamp, i);
    checkCublasError("mcd subsample kernel");


    // compute colwise means
    colwisemean_internal(d_x_subsample, p, sample_size, d_x_means);	
    checkCublasError("mcd colwise sd internal");

    // center kernel
    // center the input data columnwise 
    columnwisecenter_kernel<<< blocks, numThreads>>>(d_x_subsample, sample_size, d_x_means);
    checkCublasError("mcd colwise center internal");

    // do the matrix multiplication
    cublasDgemm('T', 'N', p, p, sample_size, 1.0, d_x_subsample, sample_size, d_x_subsample, sample_size, 0.0, d_cov, p);
    checkCublasError("mcd dgemm");

    dim3 dimGrid(p, p);	
    dim3 dimBlock(1);	
    divide_by_value_indexed_kernel<<<dimGrid,dimBlock>>>(d_cov, sample_size);
    checkCublasError("mcd divide_by_value_indexed_kernel");

    cublasGetMatrix(p, p, sizeof(MYTYPE), d_cov, p, h_covMat, p);
    checkCublasError("mcd get matrix");

    // compute determinant
    h_covMat_det[i] = compute_det(p_covMat);

    // compute inverse of subsample covariance matrix
    compute_inverse(p_covMat); 
   
    // copy inverse of covariance matrix to GPU 
    cublasSetMatrix(p, p, sizeof(MYTYPE), h_covMat, p, d_cov, p);
    
    // compute the Mahalanobis distance from each observation to the 
    // subsample center

    getNumBlocksAndThreads(kernel, sample_size, maxBlocks, maxThreads, numBlocks, numThreads);
    // mahalanobis_distance_wrapper(p, p, numThreads, numBlocks, d_x, d_x_means, d_cov, d_mh_dist);
    mahalanobis_distance_wrapper(p, p, numThreads, numBlocks, d_cov, d_mh_dist);

  }


  free(h_covMat_inverse);	

	cublasFree(d_x);
	cublasFree(d_x_means);
	cublasFree(d_x_subsample);
	cublasFree(d_sample_index);
	cublasFree(d_cov);
  
	cublasShutdown();
}

