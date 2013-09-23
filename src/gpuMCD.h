
#ifndef _robgpu_GPUMCD_H
#define _robgpu_GPUMCD_H

void gpuMCD(MYTYPE * h_X, 
            int n, 
            int p,
	          MYTYPE * h_covMat, 
            SEXP p_covMat, 
            MYTYPE * h_covMat_det,
            int nsamp,
            int sample_size,
            int * p_sample_index,
            unsigned int gpuID);

#endif // _robgpu_GPUMCD_H

