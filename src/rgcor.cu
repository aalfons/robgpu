
#include <cublas.h>
#include <cublas_v2.h>
#include <stdio.h>

#include "rgcor.h"
#include "colwisesd.h"
#include "cuseful.h"
#include "tool_kernels.h"


__host__ void rgcor(const double * x, size_t nColsx, const double * y, size_t nColsy, size_t dim, double * correlations, unsigned int gpuID)
{
  cublasStatus_t stat;
  cublasHandle_t handle;

	double *d_x = NULL;
	double *d_y = NULL;
	double *d_y_sds = NULL;
	double *d_x_sds = NULL;
	double *d_cors = NULL;
	
  cudaSetDevice(gpuID);

  stat = cublasCreate(&handle);

  if (stat != CUBLAS_STATUS_SUCCESS) {
    printf("CUBLAS initialization failed\n");
  } 
	
	// allocate device memory
	cublasAlloc(nColsx*dim, sizeof(double), (void**) &d_x);
  cublasAlloc(nColsx, sizeof(double), (void**) &d_x_sds);
	if(y != x) 
	{
		cublasAlloc(nColsy*dim, sizeof(double), (void**) &d_y);
		cublasAlloc(nColsy, sizeof(double), (void**) &d_y_sds);
		cublasAlloc(nColsx*nColsy, sizeof(double), (void**) &d_cors);
	} else {
		cublasAlloc(nColsx*nColsx, sizeof(double), (void**) &d_cors);
	}
	checkCublasError("rgcor gpu memory allocation");

	// copy input data to gpu
	cublasSetVector( nColsx*dim, sizeof(double), x, 1, d_x, 1);
	if(y != x) cublasSetVector( nColsy*dim, sizeof(double), y, 1, d_y, 1);
	
	// compute colwise sds
	colwisesd_internal(d_x, nColsx, dim, d_x_sds);	
	if(y != x) colwisesd_internal(d_y, nColsy, dim, d_y_sds);

	checkCublasError("rgcor colwisesd_internal error");

  double alpha = 1.0, beta = 0.0;
  cublasOperation_t transa = CUBLAS_OP_T, transb = CUBLAS_OP_N;

	// do the matrix multiplication
	if(y != x) 
	{
		cublasDgemm(handle, transa, transb, nColsx, nColsy, dim, &alpha, d_x, dim, d_y, dim, &beta, d_cors, nColsx);
	  checkCublasError("rgcor dgemm error");
		dim3 dimGrid(nColsx, nColsy);	
		dim3 dimBlock(1);	
		divide_by_value_indexed_kernel<<<dimGrid,dimBlock>>>(d_cors, d_x_sds, d_y_sds);
		cublasGetVector(nColsx*nColsy, sizeof(double), d_cors, 1, correlations, 1);	
	} else {
		cublasDgemm(handle, transa, transb, nColsx, nColsx, dim, &alpha, d_x, dim, d_x, dim, &beta, d_cors, nColsx);
		dim3 dimGrid(nColsx, nColsx);	
		dim3 dimBlock(1);	
		divide_by_value_indexed_kernel<<<dimGrid,dimBlock>>>(d_cors, d_x_sds, d_x_sds);
		cublasGetVector(nColsx*nColsx, sizeof(double), d_cors, 1, correlations, 1);
	}

	checkCublasError("gpuMatMult read from gpu memory");
	
	cublasFree(d_x);
	cublasFree(d_x_sds);
	if(d_y != NULL) cublasFree(d_y);	
	if(d_y_sds != NULL) cublasFree(d_y_sds);	
	cublasFree(d_cors);
	cublasDestroy(handle);
}

