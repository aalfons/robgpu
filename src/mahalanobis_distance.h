
#ifndef _robgpu_MAHALANOBIS_DISTANCE_H
#define _robgpu_MAHALANOBIS_DISTANCE_H

__global__ void mahalanobis_distance(MYTYPE * d_x, MYTYPE * d_x_means, MYTYPE * d_cov_inv, MYTYPE * d_mh_dist, int p);

#endif // _robgpu_MAHALANOBIS_DISTANCE_H

