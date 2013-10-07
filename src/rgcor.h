
#ifndef _robgpu_COR_H_
#define _robgpu_RGCOR_H_

void rgcor(const double * x, size_t nColsx,
	const double * y, size_t nColsy, size_t n, 
	double * cormat, unsigned int gpuID);

#endif // _robgpu_RGCOR_H_ 

