/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */
 
#ifndef __REDUCTION_H__
#define __REDUCTION_H__

#ifndef MIN
#define MIN(x,y) ((x < y) ? x : y)
#endif


template <class T>
void colwisesum_wrapper(int nrows, int ncols, int threads, int blocks, 
                 			T *d_idata, T *d_odata);
template <class T>
void colwisesd_wrapper(int nrows, int ncols, int threads, int blocks, 
                 			T *d_idata, T *d_odata);

template <class T>
void mahalanobis_distance_wrapper(int nrows, int ncols, int threads, int blocks, 
                 			T *d_idata, T *d_odata);

void getNumBlocksAndThreads(int whichKernel, int n, int maxBlocks, int maxThreads, int &blocks, int &threads);

extern "C" bool isPow2(unsigned int x);
unsigned int nextPow2( unsigned int x );

#endif
