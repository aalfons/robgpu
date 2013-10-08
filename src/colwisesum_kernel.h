/*
 * Code is adapted from NVIDIA examples
 *
 * See NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#ifndef _robgpu_COLWISESUM_KERNEL_H_
#define _robgpu_COLWISESUM_KERNEL_H_

template <class T>
void colwisesum_wrapper(int nrows, int ncols, int threads, int blocks, 
                 			T *d_idata, T *d_odata);

#endif // _robgpu_COLWISESUM_KERNEL_H_ 

