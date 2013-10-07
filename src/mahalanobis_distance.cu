

#include "robgpu_settings.h"
#include "mahalanobis_distance.h"

__global__ void mahalanobis_distance(MYTYPE * d_x, MYTYPE * d_x_means, MYTYPE * d_cov_inv, MYTYPE * d_mh_dist, int p)
{
  // tidx row in submatrix
  unsigned int tidx = threadIdx.x;
  // tidy column in submatrix 
  unsigned int tidy = threadIdx.y;

  // column index for the covariance matrix
  unsigned int i = (blockIdx.x + tidy) * p ;

  // row index for the covariance matrix
  unsigned int j = blockIdx.x * gridDim.x + tidx;



  // NYI geht nicht.
  d_mh_dist[i] = (d_x[i] - d_x_means[i]) * d_cov_inv[i] * (d_x[i] - d_x_means[i]);

}

