#include <R.h>
#include <Rdefines.h>

#include "fencor.h"
#include "colwisesd.h"
#include "colwisemean.h"
#include "cuseful.h"

#include<cublas.h>

// divide correlation matrix elements by sd(x)*sd(y)
__global__ void divide_by_value_indexed_kernel(double * d_cors, double * d_x_sds, double * d_y_sds)
{
	// d_cors[blockIdx.y*gridDim.x + blockIdx.x] /= ((double)(dim - 1));
    d_cors[blockIdx.y*gridDim.x + blockIdx.x] /= (d_x_sds[blockIdx.x] * d_y_sds[blockIdx.y]);	
}

__host__ void fencor(const double * x, size_t nColsx, const double * y, size_t nColsy, size_t dim, double * correlations, unsigned int gpuID)
{
	double *d_x = NULL;
	double *d_y = NULL;
	double *d_y_sds = NULL;
	double *d_x_sds = NULL;
	double *d_cors = NULL;
	
	//printf("\n ...Start of CUDA part...\n");

	// if y is NULL on R level
	//if(y == (float*)R_NilValue)
  if(nColsy == 0)
	{
		y = NULL;
	}

    cudaSetDevice(gpuID);

	cublasInit();
	
	// allocate device memory
	cublasAlloc(nColsx*dim, sizeof(double), (void**) &d_x);
    cublasAlloc(nColsx, sizeof(double), (void**) &d_x_sds);
	if(y != NULL) 
	{
		cublasAlloc(nColsy*dim, sizeof(double), (void**) &d_y);
		cublasAlloc(nColsy, sizeof(double), (void**) &d_y_sds);
		cublasAlloc(nColsx*nColsy, sizeof(double), (void**) &d_cors);
	} else {
		cublasAlloc(nColsx*nColsx, sizeof(double), (void**) &d_cors);
	}
	checkCublasError("gpuFenCor gpu memory allocation");

	// copy input data to gpu
	cublasSetVector( nColsx*dim, sizeof(double), x, 1, d_x, 1);
	if(y != NULL) cublasSetVector( nColsy*dim, sizeof(double), y, 1, d_y, 1);
	
	// compute colwise sds
	colwisesd_internal(d_x, nColsx, dim, d_x_sds);	
	if(y != NULL) colwisesd_internal(d_y, nColsy, dim, d_y_sds);

	// do the matrix multiplication
	
	if(y != NULL) 
	{
		//printf("\n vor cublasDgemm hier...\n");
		cublasDgemm('T', 'N', nColsx, nColsy, dim, 1.0, d_x, dim, d_y, dim, 0.0, d_cors, nColsx);
		dim3 dimGrid(nColsx, nColsy);	
		dim3 dimBlock(1);	
		divide_by_value_indexed_kernel<<<dimGrid,dimBlock>>>(d_cors, d_x_sds, d_y_sds);
		cublasGetVector(nColsx*nColsy, sizeof(double), d_cors, 1, correlations, 1);	
	} else {
		cublasDgemm('T', 'N', nColsx, nColsx, dim, 1.0, d_x, dim, d_x, dim, 0.0, d_cors, nColsx);
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
	cublasShutdown();
}

