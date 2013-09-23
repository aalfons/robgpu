#include "robgpu_settings.h"

#include "tools.h"

__global__ void columnwisecenter_kernel(MYTYPE * d_x, size_t n, MYTYPE * d_col_means)
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

__global__ void subsample_kernel(MYTYPE * d_x, size_t n, MYTYPE * d_x_subsample, size_t sample_size, int * d_sample_index, int nsamp, int actual_sample)
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


