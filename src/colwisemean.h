
#ifndef _robgpu_COLWISEMEAN_H_
#define _robgpu_COLWISEMEAN_H_

// expects data on host (x, means)
void colwisemean(const double * x, size_t nCols, size_t nRows, double * means);

// expects data on device (d_x, d_means)
void colwisemean_internal(double * d_x, size_t nCols, size_t nRows, double * d_means);

#endif // _robgpu_COLWISEMEAN_H_ 

