
#include "tool_kernels.h"

// TODO: OPTIMIZE!!! divide-by-nRow kernel
__global__ void divide_by_value_kernel(double * d_x, size_t n, size_t value)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	if(i < n)
		d_x[i] = d_x[i] / ((double)value);
}

// TODO: OPTIMIZE!!! sqrt_divide-by-nRow kernel
__global__ void sqrt_kernel(double * d_x, size_t n)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	if(i < n)
		d_x[i] = sqrt(d_x[i]);
}

// TODO: OPTIMIZE!!! sqrt_divide-by-nRow kernel
__global__ void sqrt_divide_by_value_kernel(double * d_x, size_t n, int value)
{
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

	if(i < n)
		d_x[i] = sqrt(d_x[i] / (double) value);
}


__global__ void columnwisecenter_kernel(double * d_x, size_t n, double * d_col_means)
{
  unsigned int blockSize = blockDim.x;
  unsigned int tid = threadIdx.x;
  unsigned int colid = blockIdx.y;
  unsigned int rowid = blockIdx.x*blockSize*2 + tid;
  unsigned int gridSize = blockSize*2*gridDim.x;

  while(rowid < n)
  {
    d_x[colid*n + rowid] -= d_col_means[colid];

    if(rowid + blockSize < n) {
      d_x[colid*n + rowid + blockSize] -= d_col_means[colid];

    }
    rowid += gridSize;
  }
}

__global__ void subsample_kernel(double * d_x, size_t n, double * d_x_subsample, size_t sample_size, int * d_sample_index, int nsamp, int actual_sample)
{
  unsigned int blockSize = blockDim.x;
  unsigned int tid = threadIdx.x;
  unsigned int colid = blockIdx.y;
  unsigned int rowid = blockIdx.x*blockSize*2 + tid;
  unsigned int gridSize = blockSize*2*gridDim.x;

  while(rowid < sample_size)
  {
    // d_x[colid*n + rowid] -= d_col_means[colid];
   
    d_x_subsample[colid * sample_size + rowid] = d_x[colid*n + d_sample_index[rowid * nsamp + actual_sample]];

    if(rowid + blockSize < sample_size) {
      // d_x[colid*n + rowid + blockSize] -= d_col_means[colid];
      d_x_subsample[colid * sample_size + rowid + blockSize] = d_x[colid*n + d_sample_index[(rowid + blockSize) * nsamp + actual_sample]];

    }

    rowid += gridSize;
  }
}

// divide correlation matrix elements by sd(x)*sd(y)
__global__ void divide_by_value_indexed_kernel(double * d_cors, double * d_x_sds, double * d_y_sds)
{
	// d_cors[blockIdx.y*gridDim.x + blockIdx.x] /= ((double)(dim - 1));
    d_cors[blockIdx.y*gridDim.x + blockIdx.x] /= (d_x_sds[blockIdx.x] * d_y_sds[blockIdx.y]);	
}





