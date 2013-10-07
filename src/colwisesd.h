
#ifndef _robgpu_COLWISESD_H_
#define _robgpu_COLWISESD_H_

// expects data on host (x, sds)
void colwisesd(const double *x, size_t nCols, size_t nRows, double * sds);

// expects data on device (d_x, d_sds)
void colwisesd_internal(double *d_x, size_t nCols, size_t nRows, double * d_sds);

#endif // _robgpu_COLWISESD_H_ 

